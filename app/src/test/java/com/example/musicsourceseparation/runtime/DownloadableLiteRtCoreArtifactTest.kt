package com.example.musicsourceseparation.runtime

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class DownloadableLiteRtCoreArtifactTest {
    @Test
    fun streamsAndHashesRuntimeEntriesWithinTheBound() {
        val source = ByteArray(256 * 1024 + 17) { index -> (index % 251).toByte() }
        val output = ByteArrayOutputStream()

        val result = copyBoundedAndHash(
            input = ByteArrayInputStream(source),
            output = output,
            maximum = source.size.toLong(),
        )

        assertEquals(source.size.toLong(), result.byteSize)
        assertArrayEquals(source, output.toByteArray())
        assertEquals(
            MessageDigest.getInstance("SHA-256").digest(source)
                .joinToString("") { byte -> "%02x".format(byte) },
            result.sha256,
        )
    }

    @Test
    fun rejectsRuntimeEntriesThatExceedTheBound() {
        assertThrows(IllegalArgumentException::class.java) {
            copyBoundedAndHash(
                input = ByteArrayInputStream(ByteArray(65 * 1024)),
                output = ByteArrayOutputStream(),
                maximum = 64 * 1024L,
            )
        }
    }

    @Test
    fun freezesExplicitLoaderReleaseContract() {
        assertEquals("2.1.5-bss.2-exp.2", DownloadableLiteRtCoreArtifact.RELEASE_VERSION)
        assertEquals(
            "https://github.com/WluhWluh/bss-litert-android/releases/download/" +
                "downloadable-runtime-v2.1.5-bss.2-exp.2/" +
                "litert-cpu-core-2.1.5-bss.2-x86.zip",
            DownloadableLiteRtCoreArtifact.forProcess(
                is64Bit = false,
                supported64BitAbis = emptyList(),
                supported32BitAbis = listOf("x86"),
            ).downloadUrl,
        )
    }

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
