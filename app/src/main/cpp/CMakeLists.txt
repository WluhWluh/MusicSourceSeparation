cmake_minimum_required(VERSION 3.22.1)

project(mss_pcm16 LANGUAGES CXX)

add_library(mss_pcm16 SHARED pcm16_jni.cpp)

target_compile_options(mss_pcm16 PRIVATE -Wall -Wextra -Werror)
