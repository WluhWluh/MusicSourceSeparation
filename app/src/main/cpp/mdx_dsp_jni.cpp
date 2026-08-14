#include <jni.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define POCKETFFT_CACHE_SIZE 8
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr int kChannels = 2;
constexpr int kComplexChannels = 4;
thread_local std::string gLastError;

int reflectIndex(int index, int size) {
    while (index < 0 || index >= size) {
        index = index < 0 ? -index : 2 * size - index - 2;
    }
    return index;
}

struct CriticalFloats {
    JNIEnv* env;
    jfloatArray array;
    jfloat* data;
    jint releaseMode;

    CriticalFloats(JNIEnv* inputEnv, jfloatArray inputArray, jint mode)
        : env(inputEnv), array(inputArray),
          data(static_cast<jfloat*>(env->GetPrimitiveArrayCritical(array, nullptr))),
          releaseMode(mode) {}
    ~CriticalFloats() {
        if (data != nullptr) env->ReleasePrimitiveArrayCritical(array, data, releaseMode);
    }
    CriticalFloats(const CriticalFloats&) = delete;
    CriticalFloats& operator=(const CriticalFloats&) = delete;
};

class MdxPlan {
public:
    MdxPlan(int fftSize, int hop, int frequencies, int frames, int samples, int workers,
            bool packedReal)
        : nFft(fftSize), hopLength(hop), dimF(frequencies), dimT(frames),
          chunkSize(samples), workerCount(workers), trim(fftSize / 2), packed(packedReal),
          window(static_cast<size_t>(fftSize)),
          windowSum(static_cast<size_t>(samples + fftSize), 0.f),
          inverseOutput(kChannels, std::vector<float>(static_cast<size_t>(samples + fftSize))),
          fftInput(static_cast<size_t>(std::max(workers, kChannels)),
                   std::vector<std::complex<float>>(static_cast<size_t>(fftSize))),
          fftOutput(static_cast<size_t>(std::max(workers, kChannels)),
                    std::vector<std::complex<float>>(static_cast<size_t>(fftSize))),
          realInput(static_cast<size_t>(std::max(workers, kChannels)),
                    std::vector<float>(static_cast<size_t>(fftSize))),
          realOutput(static_cast<size_t>(std::max(workers, kChannels)),
                     std::vector<float>(static_cast<size_t>(fftSize))),
          shape{static_cast<size_t>(fftSize)},
          stride{sizeof(std::complex<float>)}, realStride{sizeof(float)}, axes{0} {
        if (nFft <= 0 || (nFft & 1) != 0 || hopLength <= 0 || dimF <= 0 ||
            dimF > nFft / 2 + 1 || dimT <= 0 ||
            chunkSize != hopLength * (dimT - 1) || workerCount < 1 || workerCount > 8) {
            throw std::invalid_argument("invalid MDX contract");
        }
        for (int index = 0; index < nFft; ++index) {
            window[static_cast<size_t>(index)] = .5f - .5f * std::cos(
                2.f * kPi * static_cast<float>(index) / static_cast<float>(nFft));
        }
        for (int frame = 0; frame < dimT; ++frame) {
            const int start = frame * hopLength;
            for (int sample = 0; sample < nFft; ++sample) {
                const float value = window[static_cast<size_t>(sample)];
                windowSum[static_cast<size_t>(start + sample)] += value * value;
            }
        }
    }

    size_t tensorElements() const {
        return static_cast<size_t>(kComplexChannels) * dimF * dimT;
    }

    size_t channelSamples() const { return static_cast<size_t>(chunkSize); }

    void preprocess(const float* left, const float* right, float* tensor) {
        const float* channels[kChannels]{left, right};
        runWorkers(workerCount, [&](int lane) {
            auto& input = fftInput[static_cast<size_t>(lane)];
            auto& output = fftOutput[static_cast<size_t>(lane)];
            auto& real = realInput[static_cast<size_t>(lane)];
            const int firstFrame = lane * dimT / workerCount;
            const int lastFrame = (lane + 1) * dimT / workerCount;
            for (int channel = 0; channel < kChannels; ++channel) {
                for (int frame = firstFrame; frame < lastFrame; ++frame) {
                    const int frameStart = frame * hopLength - trim;
                    for (int sample = 0; sample < nFft; ++sample) {
                        const int source = reflectIndex(frameStart + sample, chunkSize);
                        const float value = channels[channel][source] * window[static_cast<size_t>(sample)];
                        input[static_cast<size_t>(sample)] = {value, 0.f};
                        real[static_cast<size_t>(sample)] = value;
                    }
                    if (packed) {
                        pocketfft::r2c(shape, realStride, stride, 0, pocketfft::FORWARD,
                                       real.data(), output.data(), 1.f, 1);
                    } else {
                        pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
                                       input.data(), output.data(), 1.f, 1);
                    }
                    const int realChannel = channel * 2;
                    for (int frequency = 0; frequency < dimF; ++frequency) {
                        const size_t base = (static_cast<size_t>(frequency) * dimT + frame) *
                            kComplexChannels;
                        tensor[base + realChannel] = output[static_cast<size_t>(frequency)].real();
                        tensor[base + realChannel + 1] = output[static_cast<size_t>(frequency)].imag();
                    }
                }
            }
        });
    }

    void postprocess(const float* tensor, float* left, float* right) {
        float* channels[kChannels]{left, right};
        runWorkers(kChannels, [&](int channel) {
            auto& input = fftInput[static_cast<size_t>(channel)];
            auto& output = fftOutput[static_cast<size_t>(channel)];
            auto& real = realOutput[static_cast<size_t>(channel)];
            auto& overlap = inverseOutput[static_cast<size_t>(channel)];
            std::fill(overlap.begin(), overlap.end(), 0.f);
            const int realChannel = channel * 2;
            for (int frame = 0; frame < dimT; ++frame) {
                std::fill(input.begin(), input.end(), std::complex<float>{0.f, 0.f});
                for (int frequency = 0; frequency < dimF; ++frequency) {
                    const size_t base = (static_cast<size_t>(frequency) * dimT + frame) *
                        kComplexChannels;
                    input[static_cast<size_t>(frequency)] = {
                        tensor[base + realChannel], tensor[base + realChannel + 1]};
                    if (!packed && frequency > 0 && frequency < nFft / 2) {
                        input[static_cast<size_t>(nFft - frequency)] =
                            std::conj(input[static_cast<size_t>(frequency)]);
                    }
                }
                if (packed) {
                    pocketfft::c2r(shape, stride, realStride, 0, pocketfft::BACKWARD,
                                   input.data(), real.data(), 1.f / static_cast<float>(nFft), 1);
                } else {
                    pocketfft::c2c(shape, stride, stride, axes, pocketfft::BACKWARD,
                                   input.data(), output.data(), 1.f / static_cast<float>(nFft), 1);
                }
                const int start = frame * hopLength;
                for (int sample = 0; sample < nFft; ++sample) {
                    overlap[static_cast<size_t>(start + sample)] +=
                        (packed ? real[static_cast<size_t>(sample)] :
                                  output[static_cast<size_t>(sample)].real()) *
                        window[static_cast<size_t>(sample)];
                }
            }
            for (int sample = 0; sample < chunkSize; ++sample) {
                const int source = sample + trim;
                channels[channel][sample] = overlap[static_cast<size_t>(source)] /
                    windowSum[static_cast<size_t>(source)];
            }
        });
    }

private:
    template <class Action>
    void runWorkers(int count, Action action) {
        if (count == 1) {
            action(0);
            return;
        }
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(count - 1));
        for (int lane = 1; lane < count; ++lane) threads.emplace_back(action, lane);
        action(0);
        for (auto& thread : threads) thread.join();
    }

    int nFft;
    int hopLength;
    int dimF;
    int dimT;
    int chunkSize;
    int workerCount;
    int trim;
    bool packed;
    std::vector<float> window;
    std::vector<float> windowSum;
    std::vector<std::vector<float>> inverseOutput;
    std::vector<std::vector<std::complex<float>>> fftInput;
    std::vector<std::vector<std::complex<float>>> fftOutput;
    std::vector<std::vector<float>> realInput;
    std::vector<std::vector<float>> realOutput;
    pocketfft::shape_t shape;
    pocketfft::stride_t stride;
    pocketfft::stride_t realStride;
    pocketfft::shape_t axes;
};

// LiteRT 2.1.5 C API profile. The ranked tensor type storage is pinned to the
// bundled 2.1.5 ABI; all other referenced objects are public opaque handles.
class LiteRt215Api {
public:
    using Handle = void*;
    using Status = int;
    using PayloadDeleter = void (*)(void*);

    LiteRt215Api() {
        library = dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) throw std::runtime_error(dlerror());
        createModelFromFile = load<CreateModelFromFile>("LiteRtCreateModelFromFile");
        destroyModel = load<DestroyHandle>("LiteRtDestroyModel");
        getModelSubgraph = load<GetIndexedHandle>("LiteRtGetModelSubgraph");
        getNumSubgraphInputs = load<GetCount>("LiteRtGetNumSubgraphInputs");
        getNumSubgraphOutputs = load<GetCount>("LiteRtGetNumSubgraphOutputs");
        getSubgraphInput = load<GetIndexedHandle>("LiteRtGetSubgraphInput");
        getSubgraphOutput = load<GetIndexedHandle>("LiteRtGetSubgraphOutput");
        getRankedTensorType = load<GetRankedTensorType>("LiteRtGetRankedTensorType");
        createOptions = load<CreateHandle>("LiteRtCreateOptions");
        destroyOptions = load<DestroyHandle>("LiteRtDestroyOptions");
        setHardwareAccelerators =
            load<SetHardwareAccelerators>("LiteRtSetOptionsHardwareAccelerators");
        addOpaqueOptions = load<AddOpaqueOptions>("LiteRtAddOpaqueOptions");
        createOpaqueOptions = load<CreateOpaqueOptions>("LiteRtCreateOpaqueOptions");
        destroyOpaqueOptions = load<DestroyHandle>("LiteRtDestroyOpaqueOptions");
        createEnvironment = load<CreateEnvironment>("LiteRtCreateEnvironment");
        destroyEnvironment = load<DestroyHandle>("LiteRtDestroyEnvironment");
        createCompiledModel = load<CreateCompiledModel>("LiteRtCreateCompiledModel");
        destroyCompiledModel = load<DestroyHandle>("LiteRtDestroyCompiledModel");
        getInputRequirements = load<GetBufferRequirements>(
            "LiteRtGetCompiledModelInputBufferRequirements");
        getOutputRequirements = load<GetBufferRequirements>(
            "LiteRtGetCompiledModelOutputBufferRequirements");
        createManagedBuffer = load<CreateManagedBuffer>(
            "LiteRtCreateManagedTensorBufferFromRequirements");
        destroyBuffer = load<DestroyHandle>("LiteRtDestroyTensorBuffer");
        getPackedSize = load<GetPackedSize>("LiteRtGetTensorBufferPackedSize");
        lockBuffer = load<LockBuffer>("LiteRtLockTensorBuffer");
        unlockBuffer = load<UnlockBuffer>("LiteRtUnlockTensorBuffer");
        runCompiledModel = load<RunCompiledModel>("LiteRtRunCompiledModel");
        getStatusString = load<GetStatusString>("LiteRtGetStatusString");
    }

    LiteRt215Api(const LiteRt215Api&) = delete;
    LiteRt215Api& operator=(const LiteRt215Api&) = delete;

    void check(Status status, const char* operation) const {
        if (status != 0) {
            const char* detail = getStatusString(status);
            throw std::runtime_error(std::string(operation) + " failed: " +
                                     (detail == nullptr ? std::to_string(status) : detail));
        }
    }

    using CreateModelFromFile = Status (*)(const char*, Handle*);
    using DestroyHandle = void (*)(Handle);
    using GetIndexedHandle = Status (*)(Handle, size_t, Handle*);
    using GetCount = Status (*)(Handle, size_t*);
    using GetRankedTensorType = Status (*)(Handle, void*);
    using CreateHandle = Status (*)(Handle*);
    using SetHardwareAccelerators = Status (*)(Handle, int);
    using AddOpaqueOptions = Status (*)(Handle, Handle);
    using CreateOpaqueOptions = Status (*)(const char*, void*, PayloadDeleter, Handle*);
    using CreateEnvironment = Status (*)(int, const void*, Handle*);
    using CreateCompiledModel = Status (*)(Handle, Handle, Handle, Handle*);
    using GetBufferRequirements = Status (*)(Handle, size_t, size_t, Handle*);
    using CreateManagedBuffer = Status (*)(Handle, const void*, Handle, Handle*);
    using GetPackedSize = Status (*)(Handle, size_t*);
    using LockBuffer = Status (*)(Handle, void**, int);
    using UnlockBuffer = Status (*)(Handle);
    using RunCompiledModel = Status (*)(Handle, size_t, size_t, Handle*, size_t, Handle*);
    using GetStatusString = const char* (*)(Status);

    CreateModelFromFile createModelFromFile;
    DestroyHandle destroyModel;
    GetIndexedHandle getModelSubgraph;
    GetCount getNumSubgraphInputs;
    GetCount getNumSubgraphOutputs;
    GetIndexedHandle getSubgraphInput;
    GetIndexedHandle getSubgraphOutput;
    GetRankedTensorType getRankedTensorType;
    CreateHandle createOptions;
    DestroyHandle destroyOptions;
    SetHardwareAccelerators setHardwareAccelerators;
    AddOpaqueOptions addOpaqueOptions;
    CreateOpaqueOptions createOpaqueOptions;
    DestroyHandle destroyOpaqueOptions;
    CreateEnvironment createEnvironment;
    DestroyHandle destroyEnvironment;
    CreateCompiledModel createCompiledModel;
    DestroyHandle destroyCompiledModel;
    GetBufferRequirements getInputRequirements;
    GetBufferRequirements getOutputRequirements;
    CreateManagedBuffer createManagedBuffer;
    DestroyHandle destroyBuffer;
    GetPackedSize getPackedSize;
    LockBuffer lockBuffer;
    UnlockBuffer unlockBuffer;
    RunCompiledModel runCompiledModel;
    GetStatusString getStatusString;

private:
    template <typename Function>
    Function load(const char* name) {
        dlerror();
        void* symbol = dlsym(library, name);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr) {
            throw std::runtime_error(std::string("Missing LiteRT 2.1.5 symbol ") + name);
        }
        return reinterpret_cast<Function>(symbol);
    }

    // The handle intentionally remains open for the process lifetime.
    void* library = nullptr;
};

LiteRt215Api& liteRtApi() {
    static LiteRt215Api api;
    return api;
}

class ScopedBufferLock {
public:
    ScopedBufferLock(LiteRt215Api& api, LiteRt215Api::Handle buffer, int write,
                     const char* operation)
        : api(api), buffer(buffer) {
        api.check(api.lockBuffer(buffer, &data, write), operation);
        locked = true;
    }

    ~ScopedBufferLock() {
        if (locked) api.unlockBuffer(buffer);
    }

    ScopedBufferLock(const ScopedBufferLock&) = delete;
    ScopedBufferLock& operator=(const ScopedBufferLock&) = delete;

    void* get() const { return data; }

    void unlock(const char* operation) {
        if (!locked) return;
        locked = false;
        api.check(api.unlockBuffer(buffer), operation);
    }

private:
    LiteRt215Api& api;
    LiteRt215Api::Handle buffer;
    void* data = nullptr;
    bool locked = false;
};

class NativeLiteRtMdxPipeline {
public:
    NativeLiteRtMdxPipeline(const char* modelPath, int fftSize, int hop,
                            int frequencies, int frames, int samples, int workers,
                            int cpuThreads, bool boundedGpu, int slotCount)
        : api(liteRtApi()),
          dsp(fftSize, hop, frequencies, frames, samples, workers, true),
          boundedGpu(boundedGpu),
          slotCount(slotCount) {
        if (cpuThreads < 1 || cpuThreads > 16) {
            throw std::invalid_argument("CPU thread count is outside 1..16");
        }
        if (slotCount < 1 || slotCount > 2) {
            throw std::invalid_argument("slot count is outside 1..2");
        }
        try {
            initialize(modelPath, cpuThreads);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~NativeLiteRtMdxPipeline() { cleanup(); }

    size_t tensorElements() const { return dsp.tensorElements(); }
    size_t channelSamples() const { return dsp.channelSamples(); }

    void preprocess(int slot, const float* left, const float* right) {
        checkSlot(slot);
        ScopedBufferLock input(api, inputBuffers[slot], 1, "LiteRtLockTensorBuffer(input)");
        dsp.preprocess(left, right, static_cast<float*>(input.get()));
        input.unlock("LiteRtUnlockTensorBuffer(input)");
    }

    void run(int slot) {
        checkSlot(slot);
        auto input = inputBuffers[slot];
        auto output = outputBuffers[slot];
        api.check(api.runCompiledModel(compiledModel, 0, 1, &input, 1, &output),
                  "LiteRtRunCompiledModel");
    }

    void postprocess(int slot, float* left, float* right) {
        checkSlot(slot);
        ScopedBufferLock output(api, outputBuffers[slot], 0, "LiteRtLockTensorBuffer(output)");
        dsp.postprocess(static_cast<const float*>(output.get()), left, right);
        output.unlock("LiteRtUnlockTensorBuffer(output)");
    }

private:
    static constexpr size_t kRankedTensorTypeBytes = 72;

    void checkSlot(int slot) const {
        if (slot < 0 || slot >= slotCount) {
            throw std::out_of_range("pipeline slot is outside configured range");
        }
    }

    void initialize(const char* modelPath, int cpuThreads) {
        api.check(api.createModelFromFile(modelPath, &model), "LiteRtCreateModelFromFile");
        api.check(api.createOptions(&options), "LiteRtCreateOptions");
        const int accelerator = boundedGpu ? 2 : 1;
        api.check(api.setHardwareAccelerators(options, accelerator),
                  boundedGpu ? "LiteRtSetOptionsHardwareAccelerators(GPU)" :
                               "LiteRtSetOptionsHardwareAccelerators(CPU)");

        const std::string identifier = boundedGpu ? "gpu_options" : "xnnpack";
        // bss.2 redirects the Kotlin command-buffer preparation field to its
        // bounded OpenCL kernel batch. C TOML bypasses that JNI redirect, so
        // retain the public field and set its effective kernel batch explicitly.
        const std::string toml = boundedGpu
            ? "backend = 1\nprecision = 2\nkernel_batch_size = 1\n"
              "num_steps_of_command_buffer_preparations = 1\n"
            : "num_threads = " + std::to_string(cpuThreads) + "\n";
        addOpaqueToml(identifier.c_str(), toml);

        api.check(api.createEnvironment(0, nullptr, &environment),
                  "LiteRtCreateEnvironment");
        api.check(api.createCompiledModel(environment, model, options, &compiledModel),
                  "LiteRtCreateCompiledModel");
        api.destroyOptions(options);
        options = nullptr;

        LiteRt215Api::Handle subgraph = nullptr;
        api.check(api.getModelSubgraph(model, 0, &subgraph), "LiteRtGetModelSubgraph");
        size_t inputCount = 0;
        size_t outputCount = 0;
        api.check(api.getNumSubgraphInputs(subgraph, &inputCount),
                  "LiteRtGetNumSubgraphInputs");
        api.check(api.getNumSubgraphOutputs(subgraph, &outputCount),
                  "LiteRtGetNumSubgraphOutputs");
        if (inputCount != 1 || outputCount != 1) {
            throw std::runtime_error("MDX LiteRT C profile requires one input and output");
        }
        inputBuffers.reserve(slotCount);
        outputBuffers.reserve(slotCount);
        for (int slot = 0; slot < slotCount; ++slot) {
            inputBuffers.push_back(createBuffer(subgraph, 0, true));
            outputBuffers.push_back(createBuffer(subgraph, 0, false));
            validatePackedSize(inputBuffers.back(), tensorElements() * sizeof(float), "input");
            validatePackedSize(outputBuffers.back(), tensorElements() * sizeof(float), "output");
        }
    }

    void addOpaqueToml(const char* identifier, const std::string& toml) {
        void* payload = std::malloc(toml.size() + 1);
        if (payload == nullptr) throw std::bad_alloc();
        std::memcpy(payload, toml.c_str(), toml.size() + 1);
        LiteRt215Api::Handle opaqueOptions = nullptr;
        const auto createStatus = api.createOpaqueOptions(
            identifier, payload, +[](void* value) { std::free(value); }, &opaqueOptions);
        if (createStatus != 0) std::free(payload);
        api.check(createStatus, "LiteRtCreateOpaqueOptions");
        const auto addStatus = api.addOpaqueOptions(options, opaqueOptions);
        if (addStatus != 0) api.destroyOpaqueOptions(opaqueOptions);
        api.check(addStatus, "LiteRtAddOpaqueOptions");
    }

    LiteRt215Api::Handle createBuffer(LiteRt215Api::Handle subgraph, size_t index,
                                      bool input) {
        LiteRt215Api::Handle tensor = nullptr;
        auto getTensor = input ? api.getSubgraphInput : api.getSubgraphOutput;
        api.check(getTensor(subgraph, index, &tensor),
                  input ? "LiteRtGetSubgraphInput" : "LiteRtGetSubgraphOutput");
        alignas(8) std::array<std::byte, kRankedTensorTypeBytes> tensorType{};
        api.check(api.getRankedTensorType(tensor, tensorType.data()),
                  "LiteRtGetRankedTensorType");
        LiteRt215Api::Handle requirements = nullptr;
        auto getRequirements = input ? api.getInputRequirements : api.getOutputRequirements;
        api.check(getRequirements(compiledModel, 0, index, &requirements),
                  input ? "LiteRtGetCompiledModelInputBufferRequirements" :
                          "LiteRtGetCompiledModelOutputBufferRequirements");
        LiteRt215Api::Handle buffer = nullptr;
        api.check(api.createManagedBuffer(environment, tensorType.data(), requirements, &buffer),
                  "LiteRtCreateManagedTensorBufferFromRequirements");
        return buffer;
    }

    void validatePackedSize(LiteRt215Api::Handle buffer, size_t expected,
                            const char* label) {
        size_t actual = 0;
        api.check(api.getPackedSize(buffer, &actual), "LiteRtGetTensorBufferPackedSize");
        if (actual != expected) {
            throw std::runtime_error(std::string(label) + " packed byte size " +
                                     std::to_string(actual) + " != " +
                                     std::to_string(expected));
        }
    }

    void cleanup() noexcept {
        for (auto buffer : inputBuffers) if (buffer != nullptr) api.destroyBuffer(buffer);
        for (auto buffer : outputBuffers) if (buffer != nullptr) api.destroyBuffer(buffer);
        if (compiledModel != nullptr) api.destroyCompiledModel(compiledModel);
        if (options != nullptr) api.destroyOptions(options);
        if (model != nullptr) api.destroyModel(model);
        if (environment != nullptr) api.destroyEnvironment(environment);
        inputBuffers.clear();
        outputBuffers.clear();
        compiledModel = nullptr;
        options = nullptr;
        model = nullptr;
        environment = nullptr;
    }

    LiteRt215Api& api;
    MdxPlan dsp;
    bool boundedGpu;
    int slotCount;
    LiteRt215Api::Handle model = nullptr;
    LiteRt215Api::Handle options = nullptr;
    LiteRt215Api::Handle environment = nullptr;
    LiteRt215Api::Handle compiledModel = nullptr;
    std::vector<LiteRt215Api::Handle> inputBuffers;
    std::vector<LiteRt215Api::Handle> outputBuffers;
};

MdxPlan* fromHandle(jlong handle) {
    return reinterpret_cast<MdxPlan*>(static_cast<intptr_t>(handle));
}

NativeLiteRtMdxPipeline* pipelineFromHandle(jlong handle) {
    return reinterpret_cast<NativeLiteRtMdxPipeline*>(static_cast<intptr_t>(handle));
}

bool hasLength(JNIEnv* env, jfloatArray array, size_t expected) {
    return array != nullptr && static_cast<size_t>(env->GetArrayLength(array)) == expected;
}

void setLastError(const std::exception& error) { gLastError = error.what(); }
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_musicsourceseparation_model_NativeMdxDsp_nativeCreate(
        JNIEnv*, jobject, jint nFft, jint hopLength, jint dimF, jint dimT,
        jint chunkSize, jint workerCount, jboolean packedReal) {
    try {
        return reinterpret_cast<jlong>(
            new MdxPlan(nFft, hopLength, dimF, dimT, chunkSize, workerCount,
                        packedReal == JNI_TRUE));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeMdxDsp_nativePreprocess(
        JNIEnv* env, jobject, jlong handle, jfloatArray leftArray,
        jfloatArray rightArray, jfloatArray tensorArray) {
    if (handle == 0 || leftArray == nullptr || rightArray == nullptr || tensorArray == nullptr) {
        return JNI_FALSE;
    }
    CriticalFloats left(env, leftArray, JNI_ABORT);
    CriticalFloats right(env, rightArray, JNI_ABORT);
    CriticalFloats tensor(env, tensorArray, 0);
    if (left.data == nullptr || right.data == nullptr || tensor.data == nullptr) return JNI_FALSE;
    fromHandle(handle)->preprocess(left.data, right.data, tensor.data);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeMdxDsp_nativePostprocess(
        JNIEnv* env, jobject, jlong handle, jfloatArray tensorArray,
        jfloatArray leftArray, jfloatArray rightArray) {
    if (handle == 0 || tensorArray == nullptr || leftArray == nullptr || rightArray == nullptr) {
        return JNI_FALSE;
    }
    CriticalFloats tensor(env, tensorArray, JNI_ABORT);
    CriticalFloats left(env, leftArray, 0);
    CriticalFloats right(env, rightArray, 0);
    if (tensor.data == nullptr || left.data == nullptr || right.data == nullptr) return JNI_FALSE;
    fromHandle(handle)->postprocess(tensor.data, left.data, right.data);
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_musicsourceseparation_model_NativeMdxDsp_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete fromHandle(handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtMdxPipeline_nativeCreate(
        JNIEnv* env, jobject, jstring modelPath, jint nFft, jint hopLength,
        jint dimF, jint dimT, jint chunkSize, jint workerCount, jint cpuThreads,
        jboolean boundedGpu, jint slotCount) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    try {
        auto* pipeline = new NativeLiteRtMdxPipeline(
            path, nFft, hopLength, dimF, dimT, chunkSize, workerCount,
            cpuThreads, boundedGpu == JNI_TRUE, slotCount);
        env->ReleaseStringUTFChars(modelPath, path);
        gLastError.clear();
        return reinterpret_cast<jlong>(pipeline);
    } catch (const std::exception& error) {
        env->ReleaseStringUTFChars(modelPath, path);
        setLastError(error);
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtMdxPipeline_nativePreprocessInput(
        JNIEnv* env, jobject, jlong handle, jint slot, jfloatArray leftArray,
        jfloatArray rightArray) {
    if (handle == 0) return JNI_FALSE;
    auto* pipeline = pipelineFromHandle(handle);
    if (!hasLength(env, leftArray, pipeline->channelSamples()) ||
        !hasLength(env, rightArray, pipeline->channelSamples())) return JNI_FALSE;
    try {
        CriticalFloats left(env, leftArray, JNI_ABORT);
        CriticalFloats right(env, rightArray, JNI_ABORT);
        if (left.data == nullptr || right.data == nullptr) return JNI_FALSE;
        pipeline->preprocess(slot, left.data, right.data);
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtMdxPipeline_nativeRun(
        JNIEnv*, jobject, jlong handle, jint slot) {
    if (handle == 0) return JNI_FALSE;
    try {
        pipelineFromHandle(handle)->run(slot);
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtMdxPipeline_nativePostprocessOutput(
        JNIEnv* env, jobject, jlong handle, jint slot, jfloatArray leftArray,
        jfloatArray rightArray) {
    if (handle == 0) return JNI_FALSE;
    auto* pipeline = pipelineFromHandle(handle);
    if (!hasLength(env, leftArray, pipeline->channelSamples()) ||
        !hasLength(env, rightArray, pipeline->channelSamples())) return JNI_FALSE;
    try {
        CriticalFloats left(env, leftArray, 0);
        CriticalFloats right(env, rightArray, 0);
        if (left.data == nullptr || right.data == nullptr) return JNI_FALSE;
        pipeline->postprocess(slot, left.data, right.data);
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtMdxPipeline_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete pipelineFromHandle(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtMdxPipeline_nativeLastError(
        JNIEnv* env, jobject) {
    return env->NewStringUTF(gLastError.c_str());
}
