package com.example.musicsourceseparation.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class DownloadableLiteRtCoreArtifactTest {
    @Test
    fun selectsTheCurrent64BitProcessAbi() {
        val arm64 = DownloadableLiteRtCoreArtifact.forProcess(
            is64Bit = true,
            supported64BitAbis = listOf("arm64-v8a"),
            supported32BitAbis = listOf("armeabi-v7a"),
        )
        val x86_64 = DownloadableLiteRtCoreArtifact.forProcess(
            is64Bit = true,
            supported64BitAbis = listOf("x86_64"),
            supported32BitAbis = listOf("x86"),
        )

        assertEquals("arm64-v8a", arm64.abi)
        assertEquals("x86_64", x86_64.abi)
    }

    @Test
    fun selectsTheCurrent32BitProcessAbi() {
        val arm32 = DownloadableLiteRtCoreArtifact.forProcess(
            is64Bit = false,
            supported64BitAbis = listOf("arm64-v8a"),
            supported32BitAbis = listOf("armeabi-v7a"),
        )
        val x86 = DownloadableLiteRtCoreArtifact.forProcess(
            is64Bit = false,
            supported64BitAbis = listOf("x86"),
            supported32BitAbis = listOf("x86"),
        )

        assertEquals("armeabi-v7a", arm32.abi)
        assertEquals("x86", x86.abi)
        assertEquals("LiteRt", x86.soname)
    }

    @Test
    fun rejectsAnUnpublishedProcessAbi() {
        assertThrows(IllegalArgumentException::class.java) {
            DownloadableLiteRtCoreArtifact.forProcess(
                is64Bit = true,
                supported64BitAbis = listOf("riscv64"),
                supported32BitAbis = emptyList(),
            )
        }
    }
}
