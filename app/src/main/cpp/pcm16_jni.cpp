#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

int16_t quantize(float value) {
    const float clipped = std::max(-1.0f, std::min(1.0f, value));
    const float scaled = clipped * 32767.0f;
    return static_cast<int16_t>(std::floor(scaled + 0.5f));
}

void encode_scalar(const float* input, int16_t* output, int count) {
    for (int index = 0; index < count; ++index) {
        output[index] = quantize(input[index]);
    }
}

#if defined(__aarch64__)
void encode_neon(const float* input, int16_t* output, int count) {
    const float32x4_t minimum = vdupq_n_f32(-1.0f);
    const float32x4_t maximum = vdupq_n_f32(1.0f);
    const float32x4_t scale = vdupq_n_f32(32767.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    int index = 0;
    for (; index + 8 <= count; index += 8) {
        float32x4_t first = vld1q_f32(input + index);
        float32x4_t second = vld1q_f32(input + index + 4);
        first = vmaxq_f32(minimum, vminq_f32(maximum, first));
        second = vmaxq_f32(minimum, vminq_f32(maximum, second));
        first = vrndmq_f32(vaddq_f32(vmulq_f32(first, scale), half));
        second = vrndmq_f32(vaddq_f32(vmulq_f32(second, scale), half));
        const int32x4_t first_int = vcvtq_s32_f32(first);
        const int32x4_t second_int = vcvtq_s32_f32(second);
        vst1q_s16(output + index, vcombine_s16(vqmovn_s32(first_int), vqmovn_s32(second_int)));
    }
    encode_scalar(input + index, output + index, count - index);
}
#endif

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_example_musicsourceseparation_audio_NativePcm16_encode(
        JNIEnv* env,
        jobject,
        jfloatArray input_array,
        jint sample_count,
        jbyteArray output_array) {
    if (sample_count < 0 || env->GetArrayLength(input_array) < sample_count ||
        env->GetArrayLength(output_array) < sample_count * 2) {
        jclass exception = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exception, "PCM16 JNI buffer is smaller than the requested sample count.");
        return 0;
    }
    auto* input = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(input_array, nullptr));
    auto* output_bytes = static_cast<jbyte*>(env->GetPrimitiveArrayCritical(output_array, nullptr));
    if (input == nullptr || output_bytes == nullptr) {
        if (input != nullptr) env->ReleasePrimitiveArrayCritical(input_array, input, JNI_ABORT);
        if (output_bytes != nullptr) {
            env->ReleasePrimitiveArrayCritical(output_array, output_bytes, 0);
        }
        jclass exception = env->FindClass("java/lang/OutOfMemoryError");
        env->ThrowNew(exception, "Could not pin PCM16 JNI buffers.");
        return 0;
    }
    auto* output = reinterpret_cast<int16_t*>(output_bytes);
#if defined(__aarch64__)
    encode_neon(input, output, sample_count);
#else
    encode_scalar(input, output, sample_count);
#endif
    env->ReleasePrimitiveArrayCritical(input_array, input, JNI_ABORT);
    env->ReleasePrimitiveArrayCritical(output_array, output_bytes, 0);
    return sample_count * 2;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_musicsourceseparation_audio_NativePcm16_implementation(
        JNIEnv* env,
        jobject) {
#if defined(__aarch64__)
    return env->NewStringUTF("neon-aarch64");
#else
    return env->NewStringUTF("scalar-native-fallback");
#endif
}
