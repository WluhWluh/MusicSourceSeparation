package com.example.musicsourceseparation.benchmark

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QnnDeviceCompatibilityTest {
    @Test
    fun preservesLiteRtSupportedQualcommDevices() {
        assertTrue(
            QnnDeviceCompatibility.isSupportedQualcommSoc(
                sdkInt = 31,
                manufacturer = "QTI",
                model = "SM8550",
                liteRtRecognized = true,
            ),
        )
    }

    @Test
    fun addsSnapdragon8Gen1And8PlusGen1() {
        assertTrue(legacyDevice("QTI", "SM8450"))
        assertTrue(legacyDevice("Qualcomm", "SM8450"))
        assertTrue(legacyDevice("QTI", "SM8475"))
        assertTrue(legacyDevice("Qualcomm", "SM8475"))
    }

    @Test
    fun addsSnapdragon7sGen2AndParrotPlatformFallback() {
        assertTrue(researchDevice("QTI", "SM7435"))
        assertTrue(researchDevice("Qualcomm", "SM7435-AB"))
        assertTrue(researchDevice("QTI", "SM7435P"))
        assertTrue(researchDevice("", "", board = "parrot"))
    }

    @Test
    fun doesNotBroadenTheAllowlistBeyondKnownResearchDevices() {
        assertFalse(legacyDevice("QTI", "SM8350"))
        assertFalse(legacyDevice("Samsung", "SM8450"))
        assertFalse(legacyDevice("QTI", "SM8450", sdkInt = 30))
        assertFalse(researchDevice("", "", board = "crow"))
    }

    private fun legacyDevice(
        manufacturer: String,
        model: String,
        sdkInt: Int = 31,
    ): Boolean = QnnDeviceCompatibility.isSupportedQualcommSoc(
        sdkInt = sdkInt,
        manufacturer = manufacturer,
        model = model,
        liteRtRecognized = false,
    )

    private fun researchDevice(
        manufacturer: String,
        model: String,
        board: String = "",
    ): Boolean = QnnDeviceCompatibility.isSupportedQualcommSoc(
        sdkInt = 31,
        manufacturer = manufacturer,
        model = model,
        board = board,
        liteRtRecognized = false,
    )
}
