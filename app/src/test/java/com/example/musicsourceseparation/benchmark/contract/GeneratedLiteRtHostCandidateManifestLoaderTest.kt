package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GeneratedLiteRtHostCandidateManifestLoaderTest {
    @Test
    fun loadsCanonicalManifestAndBindsProbeAbi() {
        val loaded = GeneratedLiteRtHostCandidateManifestLoader.load(
            manifestFile(),
            MANIFEST_BYTES,
            MANIFEST_SHA256,
        )
        val manifest = loaded.manifest

        assertEquals(MODEL_ID + "@host-1", manifest.candidateId)
        assertEquals(117_624_880L, manifest.artifact.byteSize)
        assertEquals(
            "8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7",
            manifest.artifact.sha256,
        )
        assertEquals(listOf(1, 2, 343_980), manifest.flatBuffer.inputs[0].shape)
        assertEquals(listOf(1, 4, 2_048, 336), manifest.flatBuffer.inputs[1].shape)
        assertEquals(
            listOf(4_170, 4_175),
            manifest.flatBuffer.outputs.map(ExternalLiteRtTensor::tensorIndex),
        )
        assertEquals(
            listOf("args_0", "args_1"),
            manifest.inputFixtures.mapNotNull(GeneratedLiteRtHostFixture::bindingName),
        )
        assertEquals(4_096, manifest.hostDspContract.stft.nFft)
        assertEquals(1_024, manifest.hostDspContract.stft.hopLength)
        assertEquals(347_136, manifest.hostDspContract.istft.reconstructionLength)
        assertEquals(257_985, manifest.hostDspContract.ola.strideSamples)
        assertEquals(85_995, manifest.hostDspContract.ola.overlapSamples)
        assertEquals("triangle-prefix", manifest.hostDspContract.ola.tailWeightRule)
        assertEquals(
            listOf(0, 257_985),
            manifest.hostDspContract.tailWindowPlans.map(GeneratedLiteRtHostTailWindowPlan::offset),
        )
        assertEquals(42_997, manifest.hostDspContract.tailWindowPlans.last().cropLeft)
        assertEquals(42_998, manifest.hostDspContract.tailWindowPlans.last().padRight)
        assertFalse(manifest.qualityGate.uniformTensorGatePassed)
        assertTrue(manifest.qualityGate.hostPipelineGatePassed)
        assertTrue(manifest.qualityGate.acceptedForDeviceTesting)
    }

    @Test
    fun rejectsDigestMismatch() {
        assertThrows(BenchmarkModelContractException::class.java) {
            GeneratedLiteRtHostCandidateManifestLoader.load(
                manifestFile(),
                MANIFEST_BYTES,
                "0".repeat(64),
            )
        }
    }

    @Test
    fun rejectsRemovalOfLayeredGateWarning() {
        val mutated = manifestFile().readText(Charsets.UTF_8).replaceFirst(
            "\"uniformTensorGatePassed\": false",
            "\"uniformTensorGatePassed\": true",
        )
        assertThrows(BenchmarkModelContractException::class.java) {
            GeneratedLiteRtHostCandidateManifestLoader.parse(mutated)
        }
    }

    @Test
    fun rejectsHostDspContractDrift() {
        val mutated = manifestFile().readText(Charsets.UTF_8).replaceFirst(
            "\"strideSamples\": 257985",
            "\"strideSamples\": 257984",
        )
        assertThrows(BenchmarkModelContractException::class.java) {
            GeneratedLiteRtHostCandidateManifestLoader.parse(mutated)
        }
    }

    private fun manifestFile(): File = resourceFile(
        "/benchmark-contracts/$MODEL_ID.candidate-manifest.json",
    )

    private fun resourceFile(name: String): File = File(
        requireNotNull(javaClass.getResource(name)) { "Missing test resource $name" }.toURI(),
    )

    companion object {
        private const val MODEL_ID = "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0"
        private const val MANIFEST_BYTES = 47_505L
        private const val MANIFEST_SHA256 =
            "e22708ecbb1e43f528a3f1ff2ab33a8062c42fc36ffed1134f837426865f33e2"
    }
}
