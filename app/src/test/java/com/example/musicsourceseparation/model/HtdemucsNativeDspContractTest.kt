package com.example.musicsourceseparation.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class HtdemucsNativeDspContractTest {
    @Test
    fun freezesThreePublishedModelIdentities() {
        assertEquals(
            listOf("official", "official-4s", "guitar-ft"),
            HtdemucsNativeDspContracts.ALL.map(HtdemucsNativeDspContract::variant),
        )
        assertEquals(3, HtdemucsNativeDspContracts.ALL.map { it.artifactSha256 }.toSet().size)
        assertEquals(3, HtdemucsNativeDspContracts.ALL.map { it.executableContractId }.toSet().size)
        HtdemucsNativeDspContracts.ALL.forEach { contract ->
            assertEquals(contract, HtdemucsNativeDspContracts.forVariant(contract.variant))
            assertEquals(2, contract.inputShapes.size)
            assertEquals(2, contract.outputShapes.size)
            assertEquals(listOf(1, 2, 343_980), contract.inputShapes[0])
            assertEquals(listOf(1, 4, 2_048, 336), contract.inputShapes[1])
            assertEquals(contract.sourceCount, contract.outputShapes[0][1])
            assertEquals(contract.sourceCount, contract.outputShapes[1][1])
        }
    }

    @Test
    fun officialAndGuitarFtShareDspButNotModelIdentityOrPolicy() {
        val official = HtdemucsNativeDspContracts.OFFICIAL_6S
        val guitarFt = HtdemucsNativeDspContracts.GUITAR_FT_6S
        assertEquals(official.sourceOrder, guitarFt.sourceOrder)
        assertEquals(official.inputShapes, guitarFt.inputShapes)
        assertEquals(official.outputShapes, guitarFt.outputShapes)
        assertEquals(official.normalization, guitarFt.normalization)
        assertNotEquals(official.modelId, guitarFt.modelId)
        assertNotEquals(official.artifactSha256, guitarFt.artifactSha256)
        assertFalse(official.researchOnly)
        assertTrue(guitarFt.researchOnly)
    }

    @Test
    fun fourStemContractUsesIndependentOutputAbi() {
        val contract = HtdemucsNativeDspContracts.OFFICIAL_4S
        assertEquals(listOf("drums", "bass", "other", "vocals"), contract.sourceOrder)
        assertEquals(listOf(1, 4, 4, 2_048, 336), contract.outputShapes[0])
        assertEquals(listOf(1, 4, 2, 343_980), contract.outputShapes[1])
        assertEquals(4 * 4 * 2_048 * 336, contract.frequencyOutputElements)
        assertEquals(4 * 2 * 343_980, contract.waveformOutputElements)
    }

    @Test
    fun rejectsArbitraryStemOrders() {
        assertThrows(IllegalArgumentException::class.java) {
            HtdemucsNativeDspContract(
                variant = "invalid",
                executableContractId = "invalid@1",
                modelId = "invalid",
                artifactFileName = "invalid.tflite",
                artifactByteSize = 1L,
                artifactSha256 = "0".repeat(64),
                sourceOrder = listOf("vocals"),
                researchOnly = false,
                licenseStatus = "test",
            )
        }
    }
}
