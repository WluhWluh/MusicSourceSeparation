cmake_minimum_required(VERSION 3.22.1)

project(mss_pcm16 LANGUAGES CXX)

add_library(mss_pcm16 SHARED pcm16_jni.cpp)
add_library(mss_mdx_dsp SHARED mdx_dsp_jni.cpp)

target_compile_options(mss_pcm16 PRIVATE -Wall -Wextra -Werror)
target_compile_options(mss_mdx_dsp PRIVATE -Wall -Wextra -Werror -O3)
target_include_directories(mss_mdx_dsp PRIVATE third_party/pocketfft)
