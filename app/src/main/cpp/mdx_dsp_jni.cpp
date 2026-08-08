#include <jni.h>

#include <algorithm>
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
constexpr int kComplexChannels = 4;

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

MdxPlan* fromHandle(jlong handle) {
    return reinterpret_cast<MdxPlan*>(static_cast<intptr_t>(handle));
}
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
