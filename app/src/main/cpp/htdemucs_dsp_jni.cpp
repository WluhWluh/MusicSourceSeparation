#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#define POCKETFFT_CACHE_SIZE 8
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr int kChannels = 2;
constexpr int kFeatures = 4;

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

HtdemucsPlan* fromHandle(jlong handle) {
    return reinterpret_cast<HtdemucsPlan*>(static_cast<intptr_t>(handle));
}

bool hasLength(JNIEnv* env, jfloatArray array, size_t expected) {
    return array != nullptr && static_cast<size_t>(env->GetArrayLength(array)) == expected;
}
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
