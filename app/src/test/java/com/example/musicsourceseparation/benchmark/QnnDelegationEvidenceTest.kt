package com.example.musicsourceseparation.benchmark

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QnnDelegationEvidenceTest {
    @Test
    fun acceptsNonEmptyQnnIrAsDelegationEvidence() {
        val assessment = assess(
            files = listOf(QnnIrFileEvidence("qnn_partition_0.json", 396_223L)),
        )

        assertEquals(QnnDelegationStatus.DELEGATED, assessment.status)
        assertTrue(assessment.verifiedNpuExecution)
        assertEquals(1, assessment.nonEmptyIrFileCount)
        assertEquals(396_223L, assessment.irBytes)
    }

    @Test
    fun classifiesReadyProviderWithoutIrAsCpuFallback() {
        val assessment = assess(files = emptyList())

        assertEquals(QnnDelegationStatus.CPU_FALLBACK, assessment.status)
        assertFalse(assessment.verifiedNpuExecution)
    }

    @Test
    fun keepsMissingOrMalformedEvidenceIndeterminate() {
        val missing = assess(irFilesDeclared = false, files = emptyList())
        val malformed = assess(files = emptyList(), malformedIrFileCount = 1)

        assertEquals(QnnDelegationStatus.INDETERMINATE, missing.status)
        assertEquals(QnnDelegationStatus.INDETERMINATE, malformed.status)
    }

    @Test
    fun reportsProviderPreflightFailureSeparately() {
        val assessment = QnnDelegationEvidence.assess(
            deviceSupported = true,
            libraryReady = false,
            irFilesDeclared = true,
            irFiles = emptyList(),
        )

        assertEquals(QnnDelegationStatus.PROVIDER_UNAVAILABLE, assessment.status)
        assertFalse(assessment.verifiedNpuExecution)
    }

    private fun assess(
        irFilesDeclared: Boolean = true,
        files: List<QnnIrFileEvidence>,
        malformedIrFileCount: Int = 0,
    ): QnnDelegationAssessment = QnnDelegationEvidence.assess(
        deviceSupported = true,
        libraryReady = true,
        irFilesDeclared = irFilesDeclared,
        irFiles = files,
        malformedIrFileCount = malformedIrFileCount,
    )
}
