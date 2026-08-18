#include <jni.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define POCKETFFT_CACHE_SIZE 8
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr int kChannels = 2;
constexpr int kComplexChannels = 4;
constexpr int kInputSamples = 130048;
constexpr int kNfft = 2048;
constexpr int kHopLength = 1024;
constexpr int kFrames = 128;
constexpr int kFrequencies = 1025;
constexpr int kCenterPad = kNfft / 2;
constexpr int kPaddedSamples = kInputSamples + kNfft;
constexpr int kInputElements = kInputSamples * kChannels;
constexpr int kTensorElements = kFrequencies * kFrames * kComplexChannels;
constexpr int kMaxWorkers = 4;
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

class FixedWorkerPool {
public:
    explicit FixedWorkerPool(int count) : workerCount(count) {
        for (int lane = 1; lane < workerCount; ++lane) {
            workers.emplace_back([this, lane] { workerLoop(lane); });
        }
    }

    ~FixedWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ++generation;
        }
        commandReady.notify_all();
        for (auto& worker : workers) worker.join();
    }

    FixedWorkerPool(const FixedWorkerPool&) = delete;
    FixedWorkerPool& operator=(const FixedWorkerPool&) = delete;

    void run(const std::function<void(int)>& action) {
        if (workerCount == 1) {
            action(0);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (running) throw std::logic_error("TFC-TDF worker pool is already running");
            running = true;
            currentAction = action;
            pendingWorkers = workerCount - 1;
            workerError = nullptr;
            ++generation;
        }
        commandReady.notify_all();

        std::exception_ptr callerError;
        try {
            action(0);
        } catch (...) {
            callerError = std::current_exception();
        }

        std::exception_ptr backgroundError;
        {
            std::unique_lock<std::mutex> lock(mutex);
            commandDone.wait(lock, [this] { return pendingWorkers == 0; });
            backgroundError = workerError;
            currentAction = {};
            running = false;
        }
        if (callerError != nullptr) std::rethrow_exception(callerError);
        if (backgroundError != nullptr) std::rethrow_exception(backgroundError);
    }

private:
    void workerLoop(int lane) {
        std::uint64_t observedGeneration = 0;
        while (true) {
            std::function<void(int)> action;
            {
                std::unique_lock<std::mutex> lock(mutex);
                commandReady.wait(lock, [this, observedGeneration] {
                    return stopping || generation != observedGeneration;
                });
                if (stopping) return;
                observedGeneration = generation;
                action = currentAction;
            }

            std::exception_ptr error;
            try {
                action(lane);
            } catch (...) {
                error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                if (error != nullptr && workerError == nullptr) workerError = error;
                --pendingWorkers;
                if (pendingWorkers == 0) commandDone.notify_one();
            }
        }
    }

    int workerCount;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable commandReady;
    std::condition_variable commandDone;
    std::function<void(int)> currentAction;
    std::exception_ptr workerError;
    std::uint64_t generation = 0;
    int pendingWorkers = 0;
    bool running = false;
    bool stopping = false;
};

class TfcTdfPlan {
public:
    TfcTdfPlan(int workers, bool usePackedReal)
        : workerCount(workers), packedReal(usePackedReal), pool(workers),
          window(static_cast<size_t>(kNfft)),
          windowSum(static_cast<size_t>(kPaddedSamples), 0.f),
          realInput(static_cast<size_t>(workers), std::vector<float>(kNfft)),
          complexInput(static_cast<size_t>(workers),
                       std::vector<std::complex<float>>(kNfft)),
          complexOutput(static_cast<size_t>(workers),
                        std::vector<std::complex<float>>(kNfft)),
          inverseOutput(static_cast<size_t>(workers), std::vector<float>(kNfft)),
          overlap(static_cast<size_t>(workers * kChannels),
                  std::vector<float>(kPaddedSamples)),
          shape{static_cast<size_t>(kNfft)},
          complexStride{sizeof(std::complex<float>)}, realStride{sizeof(float)}, axes{0} {
        if (workerCount < 1 || workerCount > kMaxWorkers) {
            throw std::invalid_argument("unsupported TFC-TDF worker count");
        }
        if (kInputSamples != kHopLength * (kFrames - 1) ||
            kFrequencies != kNfft / 2 + 1) {
            throw std::logic_error("invalid frozen TFC-TDF shape");
        }
        for (int index = 0; index < kNfft; ++index) {
            window[static_cast<size_t>(index)] = static_cast<float>(
                0.5 - 0.5 * std::cos(2.0 * kPi * index / kNfft));
        }
        for (int frame = 0; frame < kFrames; ++frame) {
            const int start = frame * kHopLength;
            for (int sample = 0; sample < kNfft; ++sample) {
                const float value = window[static_cast<size_t>(sample)];
                windowSum[static_cast<size_t>(start + sample)] += value * value;
            }
        }
    }

    void stft(const float* input, float* tensor) {
        pool.run([&](int lane) {
            auto& real = realInput[static_cast<size_t>(lane)];
            auto& complexIn = complexInput[static_cast<size_t>(lane)];
            auto& spectrum = complexOutput[static_cast<size_t>(lane)];
            const int firstFrame = lane * kFrames / workerCount;
            const int lastFrame = (lane + 1) * kFrames / workerCount;
            for (int channel = 0; channel < kChannels; ++channel) {
                for (int frame = firstFrame; frame < lastFrame; ++frame) {
                    const int frameStart = frame * kHopLength - kCenterPad;
                    for (int sample = 0; sample < kNfft; ++sample) {
                        const int source = reflectIndex(frameStart + sample, kInputSamples);
                        const float value = input[source * kChannels + channel] *
                            window[static_cast<size_t>(sample)];
                        real[static_cast<size_t>(sample)] = value;
                        if (!packedReal) complexIn[static_cast<size_t>(sample)] = {value, 0.f};
                    }
                    if (packedReal) {
                        pocketfft::r2c(shape, realStride, complexStride, 0,
                                       pocketfft::FORWARD, real.data(), spectrum.data(), 1.f, 1);
                    } else {
                        pocketfft::c2c(shape, complexStride, complexStride, axes,
                                       pocketfft::FORWARD, complexIn.data(), spectrum.data(), 1.f, 1);
                    }
                    for (int frequency = 0; frequency < kFrequencies; ++frequency) {
                        const size_t base =
                            (static_cast<size_t>(frequency) * kFrames + frame) * kComplexChannels;
                        tensor[base + channel] = spectrum[static_cast<size_t>(frequency)].real();
                        tensor[base + channel + kChannels] =
                            spectrum[static_cast<size_t>(frequency)].imag();
                    }
                }
            }
        });
    }

    void istft(const float* tensor, float* output) {
        reconstruct(tensor);
        pool.run([&](int lane) {
            const int firstSample = lane * kInputSamples / workerCount;
            const int lastSample = (lane + 1) * kInputSamples / workerCount;
            for (int sample = firstSample; sample < lastSample; ++sample) {
                const int paddedIndex = kCenterPad + sample;
                const float divisor = windowSum[static_cast<size_t>(paddedIndex)];
                for (int channel = 0; channel < kChannels; ++channel) {
                    output[sample * kChannels + channel] = divisor > 1e-8f
                        ? overlapValue(channel, paddedIndex) / divisor
                        : 0.f;
                }
            }
        });
    }

    void istftResidual(const float* tensor, const float* input, int trimSamples,
                       int actualSamples, float* output) {
        if (trimSamples < 0 || actualSamples < 0 ||
            trimSamples + actualSamples > kInputSamples) {
            throw std::invalid_argument("invalid TFC-TDF residual crop");
        }
        reconstruct(tensor);
        pool.run([&](int lane) {
            const int firstSample = lane * actualSamples / workerCount;
            const int lastSample = (lane + 1) * actualSamples / workerCount;
            for (int sample = firstSample; sample < lastSample; ++sample) {
                const int sourceSample = trimSamples + sample;
                const int paddedIndex = kCenterPad + sourceSample;
                const float divisor = windowSum[static_cast<size_t>(paddedIndex)];
                for (int channel = 0; channel < kChannels; ++channel) {
                    const float reconstructed = divisor > 1e-8f
                        ? overlapValue(channel, paddedIndex) / divisor
                        : 0.f;
                    output[sample * kChannels + channel] =
                        input[sourceSample * kChannels + channel] - reconstructed;
                }
            }
        });
    }

private:
    void reconstruct(const float* tensor) {
        pool.run([&](int lane) {
            auto& spectrum = complexInput[static_cast<size_t>(lane)];
            auto& complexInverse = complexOutput[static_cast<size_t>(lane)];
            auto& realInverse = inverseOutput[static_cast<size_t>(lane)];
            for (int channel = 0; channel < kChannels; ++channel) {
                auto& laneOverlap = overlap[overlapIndex(lane, channel)];
                std::fill(laneOverlap.begin(), laneOverlap.end(), 0.f);
            }
            const int firstFrame = lane * kFrames / workerCount;
            const int lastFrame = (lane + 1) * kFrames / workerCount;
            for (int channel = 0; channel < kChannels; ++channel) {
                auto& laneOverlap = overlap[overlapIndex(lane, channel)];
                for (int frame = firstFrame; frame < lastFrame; ++frame) {
                    if (!packedReal) {
                        std::fill(spectrum.begin(), spectrum.end(),
                                  std::complex<float>{0.f, 0.f});
                    }
                    for (int frequency = 0; frequency < kFrequencies; ++frequency) {
                        const size_t base =
                            (static_cast<size_t>(frequency) * kFrames + frame) * kComplexChannels;
                        std::complex<float> value{
                            tensor[base + channel],
                            tensor[base + channel + kChannels],
                        };
                        if (frequency == 0 || frequency == kFrequencies - 1) {
                            value.imag(0.f);
                        }
                        spectrum[static_cast<size_t>(frequency)] = value;
                        if (!packedReal && frequency > 0 && frequency < kFrequencies - 1) {
                            spectrum[static_cast<size_t>(kNfft - frequency)] = std::conj(value);
                        }
                    }
                    if (packedReal) {
                        pocketfft::c2r(shape, complexStride, realStride, 0,
                                       pocketfft::BACKWARD, spectrum.data(), realInverse.data(),
                                       1.f / static_cast<float>(kNfft), 1);
                    } else {
                        pocketfft::c2c(shape, complexStride, complexStride, axes,
                                       pocketfft::BACKWARD, spectrum.data(), complexInverse.data(),
                                       1.f / static_cast<float>(kNfft), 1);
                    }
                    const int start = frame * kHopLength;
                    for (int sample = 0; sample < kNfft; ++sample) {
                        const float value = packedReal
                            ? realInverse[static_cast<size_t>(sample)]
                            : complexInverse[static_cast<size_t>(sample)].real();
                        laneOverlap[static_cast<size_t>(start + sample)] +=
                            value * window[static_cast<size_t>(sample)];
                    }
                }
            }
        });
    }

    float overlapValue(int channel, int paddedIndex) const {
        float value = 0.f;
        for (int lane = 0; lane < workerCount; ++lane) {
            value += overlap[overlapIndex(lane, channel)][static_cast<size_t>(paddedIndex)];
        }
        return value;
    }

    static size_t overlapIndex(int lane, int channel) {
        return static_cast<size_t>(lane * kChannels + channel);
    }

    int workerCount;
    bool packedReal;
    FixedWorkerPool pool;
    std::vector<float> window;
    std::vector<float> windowSum;
    std::vector<std::vector<float>> realInput;
    std::vector<std::vector<std::complex<float>>> complexInput;
    std::vector<std::vector<std::complex<float>>> complexOutput;
    std::vector<std::vector<float>> inverseOutput;
    std::vector<std::vector<float>> overlap;
    pocketfft::shape_t shape;
    pocketfft::stride_t complexStride;
    pocketfft::stride_t realStride;
    pocketfft::shape_t axes;
};

TfcTdfPlan* fromHandle(jlong handle) {
    if (handle == 0) throw std::invalid_argument("TFC-TDF DSP handle is null");
    return reinterpret_cast<TfcTdfPlan*>(handle);
}

bool checkLength(JNIEnv* env, jfloatArray array, int expected, const char* label) {
    if (array == nullptr || env->GetArrayLength(array) != expected) {
        gLastError = std::string(label) + " length mismatch";
        return false;
    }
    return true;
}
}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_musicsourceseparation_model_NativeTfcTdfDsp_nativeCreate(
    JNIEnv*, jobject, jint workerCount, jboolean packedReal) {
    try {
        gLastError.clear();
        return reinterpret_cast<jlong>(
            new TfcTdfPlan(workerCount, packedReal == JNI_TRUE));
    } catch (const std::exception& error) {
        gLastError = error.what();
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeTfcTdfDsp_nativeStft(
    JNIEnv* env, jobject, jlong handle, jfloatArray inputArray, jfloatArray tensorArray) {
    try {
        gLastError.clear();
        if (!checkLength(env, inputArray, kInputElements, "input") ||
            !checkLength(env, tensorArray, kTensorElements, "tensor")) {
            return JNI_FALSE;
        }
        CriticalFloats input(env, inputArray, JNI_ABORT);
        CriticalFloats tensor(env, tensorArray, 0);
        if (input.data == nullptr || tensor.data == nullptr) {
            gLastError = "unable to pin TFC-TDF STFT arrays";
            return JNI_FALSE;
        }
        fromHandle(handle)->stft(input.data, tensor.data);
        return JNI_TRUE;
    } catch (const std::exception& error) {
        gLastError = error.what();
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeTfcTdfDsp_nativeIstft(
    JNIEnv* env, jobject, jlong handle, jfloatArray tensorArray, jfloatArray outputArray) {
    try {
        gLastError.clear();
        if (!checkLength(env, tensorArray, kTensorElements, "tensor") ||
            !checkLength(env, outputArray, kInputElements, "output")) {
            return JNI_FALSE;
        }
        CriticalFloats tensor(env, tensorArray, JNI_ABORT);
        CriticalFloats output(env, outputArray, 0);
        if (tensor.data == nullptr || output.data == nullptr) {
            gLastError = "unable to pin TFC-TDF iSTFT arrays";
            return JNI_FALSE;
        }
        fromHandle(handle)->istft(tensor.data, output.data);
        return JNI_TRUE;
    } catch (const std::exception& error) {
        gLastError = error.what();
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_musicsourceseparation_model_NativeTfcTdfDsp_nativeIstftResidual(
    JNIEnv* env, jobject, jlong handle, jfloatArray tensorArray, jfloatArray inputArray,
    jint trimSamples, jint actualSamples, jfloatArray outputArray) {
    try {
        gLastError.clear();
        if (!checkLength(env, tensorArray, kTensorElements, "tensor") ||
            !checkLength(env, inputArray, kInputElements, "input") ||
            !checkLength(env, outputArray, actualSamples * kChannels, "output")) {
            return JNI_FALSE;
        }
        CriticalFloats tensor(env, tensorArray, JNI_ABORT);
        CriticalFloats input(env, inputArray, JNI_ABORT);
        CriticalFloats output(env, outputArray, 0);
        if (tensor.data == nullptr || input.data == nullptr || output.data == nullptr) {
            gLastError = "unable to pin TFC-TDF residual arrays";
            return JNI_FALSE;
        }
        fromHandle(handle)->istftResidual(
            tensor.data, input.data, trimSamples, actualSamples, output.data);
        return JNI_TRUE;
    } catch (const std::exception& error) {
        gLastError = error.what();
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_musicsourceseparation_model_NativeTfcTdfDsp_nativeDestroy(
    JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<TfcTdfPlan*>(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_musicsourceseparation_model_NativeTfcTdfDsp_nativeLastError(
    JNIEnv* env, jobject) {
    return env->NewStringUTF(gLastError.c_str());
}
