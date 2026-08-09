#include <jni.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
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
constexpr int kFeatures = 4;
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
        : env(inputEnv), array(inputArray), data(inputArray == nullptr ? nullptr :
              static_cast<jfloat*>(env->GetPrimitiveArrayCritical(inputArray, nullptr))),
          releaseMode(mode) {}
    ~CriticalFloats() {
        if (data != nullptr) env->ReleasePrimitiveArrayCritical(array, data, releaseMode);
    }
    CriticalFloats(const CriticalFloats&) = delete;
    CriticalFloats& operator=(const CriticalFloats&) = delete;
};

class HtdemucsPlan {
public:
    HtdemucsPlan(int sources, int samples, int frames, int fftSize, int hop,
                 int frequencies, int outerLeft, int leftFrames, int rightFrames,
                 int trim, int workers)
        : sourceCount(sources), windowSamples(samples), spectrumFrames(frames),
          nFft(fftSize), hopSize(hop), dimF(frequencies), outerPadLeft(outerLeft),
          framePadLeft(leftFrames), framePadRight(rightFrames), centerTrim(trim),
          workerCount(workers), outerLength(frames * hop + 2 * outerLeft),
          fullFrameCount(frames + leftFrames + rightFrames),
          reconstructedLength((fullFrameCount - 1) * hop + fftSize),
          cropStart(trim + outerLeft), window(static_cast<size_t>(fftSize)),
          inverseEnvelope(static_cast<size_t>(reconstructedLength), 0.f),
          shape{static_cast<size_t>(fftSize)},
          complexStride{sizeof(std::complex<float>)}, realStride{sizeof(float)} {
        if ((sourceCount != 4 && sourceCount != 6) || windowSamples != 343980 ||
            spectrumFrames != 336 || nFft != 4096 || hopSize != 1024 || dimF != 2048 ||
            outerPadLeft != 1536 || framePadLeft != 2 || framePadRight != 2 ||
            centerTrim != 2048 || workerCount < 1 || workerCount > 8 ||
            spectrumFrames != (windowSamples + hopSize - 1) / hopSize ||
            cropStart + windowSamples > reconstructedLength) {
            throw std::invalid_argument("unsupported HTDemucs DSP contract");
        }
        for (int index = 0; index < nFft; ++index) {
            window[static_cast<size_t>(index)] = .5f - .5f * std::cos(
                2.f * kPi * static_cast<float>(index) / static_cast<float>(nFft));
        }
        for (int frame = 0; frame < fullFrameCount; ++frame) {
            const int offset = frame * hopSize;
            for (int sample = 0; sample < nFft; ++sample) {
                const float value = window[static_cast<size_t>(sample)];
                inverseEnvelope[static_cast<size_t>(offset + sample)] += value * value;
            }
        }
        for (float& value : inverseEnvelope) value = value > 1e-10f ? 1.f / value : 0.f;
    }

    size_t waveformInputElements() const {
        return static_cast<size_t>(kChannels) * windowSamples;
    }
    size_t spectrumInputElements() const {
        return static_cast<size_t>(kFeatures) * dimF * spectrumFrames;
    }
    size_t frequencyOutputElements() const {
        return static_cast<size_t>(sourceCount) * spectrumInputElements();
    }
    size_t waveformOutputElements() const {
        return static_cast<size_t>(sourceCount) * kChannels * windowSamples;
    }

    void preprocess(const float* waveform, float* spectrum) const {
        const float scale = 1.f / std::sqrt(static_cast<float>(nFft));
        runWorkers(kChannels, [&](int channel) {
            std::vector<float> input(static_cast<size_t>(nFft));
            std::vector<std::complex<float>> output(static_cast<size_t>(nFft / 2 + 1));
            for (int frame = 0; frame < spectrumFrames; ++frame) {
                const int outerStart = (frame + framePadLeft) * hopSize - centerTrim;
                for (int sample = 0; sample < nFft; ++sample) {
                    const int outerIndex = reflectIndex(outerStart + sample, outerLength);
                    const int rawIndex = reflectIndex(outerIndex - outerPadLeft, windowSamples);
                    input[static_cast<size_t>(sample)] =
                        waveform[static_cast<size_t>(channel) * windowSamples + rawIndex] *
                        window[static_cast<size_t>(sample)];
                }
                pocketfft::r2c(shape, realStride, complexStride, 0, pocketfft::FORWARD,
                               input.data(), output.data(), scale, 1);
                const int realFeature = channel * 2;
                for (int frequency = 0; frequency < dimF; ++frequency) {
                    const size_t realIndex =
                        (static_cast<size_t>(realFeature) * dimF + frequency) *
                        spectrumFrames + frame;
                    spectrum[realIndex] = output[static_cast<size_t>(frequency)].real();
                    spectrum[realIndex + static_cast<size_t>(dimF) * spectrumFrames] =
                        output[static_cast<size_t>(frequency)].imag();
                }
            }
        });
    }

    void postprocess(const float* frequency, const float* timeWaveform, float* output) const {
        std::atomic<int> nextPlane{0};
        const int planeCount = sourceCount * kChannels;
        const float inverseScale = std::sqrt(static_cast<float>(nFft)) /
            static_cast<float>(nFft);
        runWorkers(std::min(workerCount, planeCount), [&](int) {
            std::vector<std::complex<float>> input(static_cast<size_t>(nFft / 2 + 1));
            std::vector<float> inverse(static_cast<size_t>(nFft));
            std::vector<float> reconstructed(static_cast<size_t>(reconstructedLength));
            while (true) {
                const int plane = nextPlane.fetch_add(1, std::memory_order_relaxed);
                if (plane >= planeCount) return;
                const int source = plane / kChannels;
                const int channel = plane % kChannels;
                std::fill(reconstructed.begin(), reconstructed.end(), 0.f);
                for (int fullFrame = 0; fullFrame < fullFrameCount; ++fullFrame) {
                    std::fill(input.begin(), input.end(), std::complex<float>{0.f, 0.f});
                    if (fullFrame >= framePadLeft &&
                        fullFrame < spectrumFrames + framePadLeft) {
                        const int modelFrame = fullFrame - framePadLeft;
                        const int realFeature = source * kFeatures + channel * 2;
                        for (int bin = 0; bin < dimF; ++bin) {
                            const size_t realIndex =
                                (static_cast<size_t>(realFeature) * dimF + bin) *
                                spectrumFrames + modelFrame;
                            input[static_cast<size_t>(bin)] = {
                                frequency[realIndex],
                                frequency[realIndex + static_cast<size_t>(dimF) * spectrumFrames],
                            };
                        }
                    }
                    pocketfft::c2r(shape, complexStride, realStride, 0, pocketfft::BACKWARD,
                                   input.data(), inverse.data(), inverseScale, 1);
                    const int offset = fullFrame * hopSize;
                    for (int sample = 0; sample < nFft; ++sample) {
                        reconstructed[static_cast<size_t>(offset + sample)] +=
                            inverse[static_cast<size_t>(sample)] * window[static_cast<size_t>(sample)];
                    }
                }
                const size_t base = static_cast<size_t>(plane) * windowSamples;
                for (int sample = 0; sample < windowSamples; ++sample) {
                    const int index = cropStart + sample;
                    const float frequencyValue = reconstructed[static_cast<size_t>(index)] *
                        inverseEnvelope[static_cast<size_t>(index)];
                    output[base + sample] = frequencyValue +
                        (timeWaveform == nullptr ? 0.f : timeWaveform[base + sample]);
                }
            }
        });
    }

private:
    template <class Action>
    void runWorkers(int count, Action action) const {
        if (count == 1) {
            action(0);
            return;
        }
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(count - 1));
        for (int worker = 1; worker < count; ++worker) threads.emplace_back(action, worker);
        action(0);
        for (auto& thread : threads) thread.join();
    }

    int sourceCount;
    int windowSamples;
    int spectrumFrames;
    int nFft;
    int hopSize;
    int dimF;
    int outerPadLeft;
    int framePadLeft;
    int framePadRight;
    int centerTrim;
    int workerCount;
    int outerLength;
    int fullFrameCount;
    int reconstructedLength;
    int cropStart;
    std::vector<float> window;
    std::vector<float> inverseEnvelope;
    pocketfft::shape_t shape;
    pocketfft::stride_t complexStride;
    pocketfft::stride_t realStride;
};

// LiteRT 2.1.5 C API profile. All referenced types are opaque except the ranked
// tensor type, whose storage size is pinned to the bundled 2.1.5 ABI.
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

    void* library = nullptr;
};

LiteRt215Api& liteRtApi() {
    // Keep libLiteRt loaded for the process lifetime so cached function pointers remain valid.
    static LiteRt215Api api;
    return api;
}

class NativeLiteRtPipeline {
public:
    NativeLiteRtPipeline(const char* modelPath, int sources, int samples, int frames,
                         int fftSize, int hop, int frequencies, int outerLeft,
                         int leftFrames, int rightFrames, int trim, int cpuThreads,
                         int workers)
        : api(liteRtApi()),
          dsp(sources, samples, frames, fftSize, hop, frequencies, outerLeft,
              leftFrames, rightFrames, trim, workers) {
        if (cpuThreads < 1 || cpuThreads > 16) {
            throw std::invalid_argument("CPU thread count is outside 1..16");
        }
        try {
            initialize(modelPath, cpuThreads);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~NativeLiteRtPipeline() { cleanup(); }

    size_t waveformInputElements() const { return dsp.waveformInputElements(); }
    size_t spectrumInputElements() const { return dsp.spectrumInputElements(); }
    size_t frequencyOutputElements() const { return dsp.frequencyOutputElements(); }
    size_t waveformOutputElements() const { return dsp.waveformOutputElements(); }

    void preprocess(const float* waveform, float* spectrum) const {
        dsp.preprocess(waveform, spectrum);
    }

    void writeInputs(const float* waveform, const float* spectrum) {
        writeBuffer(inputBuffers[0], waveform, waveformInputElements() * sizeof(float));
        writeBuffer(inputBuffers[1], spectrum, spectrumInputElements() * sizeof(float));
    }

    void run() {
        api.check(api.runCompiledModel(compiledModel, 0, inputBuffers.size(),
                                       inputBuffers.data(), outputBuffers.size(),
                                       outputBuffers.data()),
                  "LiteRtRunCompiledModel");
    }

    void postprocess(float* output) {
        ScopedBufferLock frequency(api, outputBuffers[0], 0,
                                   "LiteRtLockTensorBuffer(frequency)");
        ScopedBufferLock time(api, outputBuffers[1], 0,
                              "LiteRtLockTensorBuffer(time)");
        dsp.postprocess(static_cast<const float*>(frequency.get()),
                        static_cast<const float*>(time.get()), output);
        time.unlock("LiteRtUnlockTensorBuffer(time)");
        frequency.unlock("LiteRtUnlockTensorBuffer(frequency)");
    }

private:
    static constexpr size_t kRankedTensorTypeBytes = 72;

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

    void initialize(const char* modelPath, int cpuThreads) {
        api.check(api.createModelFromFile(modelPath, &model), "LiteRtCreateModelFromFile");
        api.check(api.createOptions(&options), "LiteRtCreateOptions");
        api.check(api.setHardwareAccelerators(options, 1),
                  "LiteRtSetOptionsHardwareAccelerators(CPU)");

        const std::string cpuToml = "num_threads = " + std::to_string(cpuThreads) + "\n";
        void* cpuPayload = std::malloc(cpuToml.size() + 1);
        if (cpuPayload == nullptr) throw std::bad_alloc();
        std::memcpy(cpuPayload, cpuToml.c_str(), cpuToml.size() + 1);
        LiteRt215Api::Handle opaqueOptions = nullptr;
        const auto createOpaqueStatus = api.createOpaqueOptions(
            "xnnpack", cpuPayload, +[](void* payload) { std::free(payload); },
            &opaqueOptions);
        if (createOpaqueStatus != 0) std::free(cpuPayload);
        api.check(createOpaqueStatus, "LiteRtCreateOpaqueOptions(CPU)");
        const auto addStatus = api.addOpaqueOptions(options, opaqueOptions);
        if (addStatus != 0) api.destroyOpaqueOptions(opaqueOptions);
        api.check(addStatus, "LiteRtAddOpaqueOptions(CPU)");

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
        if (inputCount != 2 || outputCount != 2) {
            throw std::runtime_error("HTDemucs LiteRT C profile requires two inputs and outputs");
        }
        inputBuffers = createBuffers(subgraph, inputCount, true);
        outputBuffers = createBuffers(subgraph, outputCount, false);
        validatePackedSize(inputBuffers[0], waveformInputElements() * sizeof(float),
                           "waveform input");
        validatePackedSize(inputBuffers[1], spectrumInputElements() * sizeof(float),
                           "spectrum input");
        validatePackedSize(outputBuffers[0], frequencyOutputElements() * sizeof(float),
                           "frequency output");
        validatePackedSize(outputBuffers[1], waveformOutputElements() * sizeof(float),
                           "time output");
    }

    std::vector<LiteRt215Api::Handle> createBuffers(LiteRt215Api::Handle subgraph,
                                                     size_t count, bool input) {
        std::vector<LiteRt215Api::Handle> result;
        result.reserve(count);
        try {
            for (size_t index = 0; index < count; ++index) {
                LiteRt215Api::Handle tensor = nullptr;
                auto getTensor = input ? api.getSubgraphInput : api.getSubgraphOutput;
                api.check(getTensor(subgraph, index, &tensor),
                          input ? "LiteRtGetSubgraphInput" : "LiteRtGetSubgraphOutput");
                alignas(8) std::array<std::byte, kRankedTensorTypeBytes> tensorType{};
                api.check(api.getRankedTensorType(tensor, tensorType.data()),
                          "LiteRtGetRankedTensorType");
                LiteRt215Api::Handle requirements = nullptr;
                auto getRequirements =
                    input ? api.getInputRequirements : api.getOutputRequirements;
                api.check(getRequirements(compiledModel, 0, index, &requirements),
                          input ? "LiteRtGetCompiledModelInputBufferRequirements" :
                                  "LiteRtGetCompiledModelOutputBufferRequirements");
                LiteRt215Api::Handle buffer = nullptr;
                api.check(api.createManagedBuffer(environment, tensorType.data(), requirements,
                                                  &buffer),
                          "LiteRtCreateManagedTensorBufferFromRequirements");
                result.push_back(buffer);
            }
        } catch (...) {
            for (auto buffer : result) api.destroyBuffer(buffer);
            throw;
        }
        return result;
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

    void writeBuffer(LiteRt215Api::Handle buffer, const float* source, size_t bytes) {
        ScopedBufferLock destination(api, buffer, 1, "LiteRtLockTensorBuffer(write)");
        std::memcpy(destination.get(), source, bytes);
        destination.unlock("LiteRtUnlockTensorBuffer(write)");
    }

    void cleanup() noexcept {
        for (auto buffer : inputBuffers) if (buffer != nullptr) api.destroyBuffer(buffer);
        for (auto buffer : outputBuffers) if (buffer != nullptr) api.destroyBuffer(buffer);
        inputBuffers.clear();
        outputBuffers.clear();
        if (compiledModel != nullptr) api.destroyCompiledModel(compiledModel);
        if (options != nullptr) api.destroyOptions(options);
        if (model != nullptr) api.destroyModel(model);
        if (environment != nullptr) api.destroyEnvironment(environment);
        compiledModel = nullptr;
        options = nullptr;
        model = nullptr;
        environment = nullptr;
    }

    LiteRt215Api& api;
    HtdemucsPlan dsp;
    LiteRt215Api::Handle model = nullptr;
    LiteRt215Api::Handle options = nullptr;
    LiteRt215Api::Handle environment = nullptr;
    LiteRt215Api::Handle compiledModel = nullptr;
    std::vector<LiteRt215Api::Handle> inputBuffers;
    std::vector<LiteRt215Api::Handle> outputBuffers;
};

HtdemucsPlan* fromHandle(jlong handle) {
    return reinterpret_cast<HtdemucsPlan*>(static_cast<intptr_t>(handle));
}

bool hasLength(JNIEnv* env, jfloatArray array, size_t expected) {
    return array != nullptr && static_cast<size_t>(env->GetArrayLength(array)) == expected;
}

NativeLiteRtPipeline* pipelineFromHandle(jlong handle) {
    return reinterpret_cast<NativeLiteRtPipeline*>(static_cast<intptr_t>(handle));
}

void setLastError(const std::exception& error) { gLastError = error.what(); }
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_musicsourceseparation_model_NativeHtdemucsDsp_nativeCreate(
        JNIEnv*, jobject, jint sourceCount, jint windowSamples, jint spectrumFrames,
        jint fftSize, jint hopSize, jint dimF, jint outerPadLeft, jint framePadLeft,
        jint framePadRight, jint centerTrim, jint workerCount) {
    try {
        return reinterpret_cast<jlong>(new HtdemucsPlan(
            sourceCount, windowSamples, spectrumFrames, fftSize, hopSize, dimF,
            outerPadLeft, framePadLeft, framePadRight, centerTrim, workerCount));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeHtdemucsDsp_nativePreprocess(
        JNIEnv* env, jobject, jlong handle, jfloatArray waveformArray,
        jfloatArray spectrumArray) {
    if (handle == 0) return JNI_FALSE;
    const auto* plan = fromHandle(handle);
    if (!hasLength(env, waveformArray, plan->waveformInputElements()) ||
        !hasLength(env, spectrumArray, plan->spectrumInputElements())) return JNI_FALSE;
    CriticalFloats waveform(env, waveformArray, JNI_ABORT);
    CriticalFloats spectrum(env, spectrumArray, 0);
    if (waveform.data == nullptr || spectrum.data == nullptr) return JNI_FALSE;
    plan->preprocess(waveform.data, spectrum.data);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeHtdemucsDsp_nativePostprocess(
        JNIEnv* env, jobject, jlong handle, jfloatArray frequencyArray,
        jfloatArray timeArray, jfloatArray outputArray) {
    if (handle == 0) return JNI_FALSE;
    const auto* plan = fromHandle(handle);
    if (!hasLength(env, frequencyArray, plan->frequencyOutputElements()) ||
        (timeArray != nullptr && !hasLength(env, timeArray, plan->waveformOutputElements())) ||
        !hasLength(env, outputArray, plan->waveformOutputElements())) return JNI_FALSE;
    CriticalFloats frequency(env, frequencyArray, JNI_ABORT);
    CriticalFloats time(env, timeArray, JNI_ABORT);
    CriticalFloats output(env, outputArray, 0);
    if (frequency.data == nullptr || output.data == nullptr) return JNI_FALSE;
    plan->postprocess(frequency.data, time.data, output.data);
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_musicsourceseparation_model_NativeHtdemucsDsp_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete fromHandle(handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativeCreate(
        JNIEnv* env, jobject, jstring modelPath, jint sourceCount, jint windowSamples,
        jint spectrumFrames, jint fftSize, jint hopSize, jint dimF, jint outerPadLeft,
        jint framePadLeft, jint framePadRight, jint centerTrim, jint cpuThreads,
        jint workerCount) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (path == nullptr) return 0;
    try {
        auto* pipeline = new NativeLiteRtPipeline(
            path, sourceCount, windowSamples, spectrumFrames, fftSize, hopSize, dimF,
            outerPadLeft, framePadLeft, framePadRight, centerTrim, cpuThreads, workerCount);
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
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativePreprocess(
        JNIEnv* env, jobject, jlong handle, jfloatArray waveformArray,
        jfloatArray spectrumArray) {
    if (handle == 0) return JNI_FALSE;
    auto* pipeline = pipelineFromHandle(handle);
    if (!hasLength(env, waveformArray, pipeline->waveformInputElements()) ||
        !hasLength(env, spectrumArray, pipeline->spectrumInputElements())) return JNI_FALSE;
    try {
        CriticalFloats waveform(env, waveformArray, JNI_ABORT);
        CriticalFloats spectrum(env, spectrumArray, 0);
        if (waveform.data == nullptr || spectrum.data == nullptr) return JNI_FALSE;
        pipeline->preprocess(waveform.data, spectrum.data);
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativeWriteInputs(
        JNIEnv* env, jobject, jlong handle, jfloatArray waveformArray,
        jfloatArray spectrumArray) {
    if (handle == 0) return JNI_FALSE;
    auto* pipeline = pipelineFromHandle(handle);
    if (!hasLength(env, waveformArray, pipeline->waveformInputElements()) ||
        !hasLength(env, spectrumArray, pipeline->spectrumInputElements())) return JNI_FALSE;
    try {
        CriticalFloats waveform(env, waveformArray, JNI_ABORT);
        CriticalFloats spectrum(env, spectrumArray, JNI_ABORT);
        if (waveform.data == nullptr || spectrum.data == nullptr) return JNI_FALSE;
        pipeline->writeInputs(waveform.data, spectrum.data);
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativeRun(
        JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return JNI_FALSE;
    try {
        pipelineFromHandle(handle)->run();
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativePostprocessOutputs(
        JNIEnv* env, jobject, jlong handle, jfloatArray outputArray) {
    if (handle == 0) return JNI_FALSE;
    auto* pipeline = pipelineFromHandle(handle);
    if (!hasLength(env, outputArray, pipeline->waveformOutputElements())) return JNI_FALSE;
    try {
        CriticalFloats output(env, outputArray, 0);
        if (output.data == nullptr) return JNI_FALSE;
        pipeline->postprocess(output.data);
        gLastError.clear();
        return JNI_TRUE;
    } catch (const std::exception& error) {
        setLastError(error);
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativeDestroy(
        JNIEnv*, jobject, jlong handle) {
    delete pipelineFromHandle(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_musicsourceseparation_model_NativeLiteRtHtdemucsPipeline_nativeLastError(
        JNIEnv* env, jobject) {
    return env->NewStringUTF(gLastError.c_str());
}
