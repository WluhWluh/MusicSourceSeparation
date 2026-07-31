package com.example.musicsourceseparation.benchmark

import android.os.Build
import com.google.ai.edge.litert.NpuCompatibilityChecker
import java.util.Locale

internal object QnnDeviceCompatibility {
    const val CHECKER_NAME = "QualcommWithResearchSocAllowlist"

    private val additionalSupportedSocs = setOf(
        QualcommSoc("QTI", "SM8450"),
        QualcommSoc("Qualcomm", "SM8450"),
        QualcommSoc("QTI", "SM8475"),
        QualcommSoc("Qualcomm", "SM8475"),
        QualcommSoc("QTI", "SM7435"),
        QualcommSoc("Qualcomm", "SM7435"),
        QualcommSoc("QTI", "SM7435-AB"),
        QualcommSoc("Qualcomm", "SM7435-AB"),
        QualcommSoc("QTI", "SM7435P"),
        QualcommSoc("Qualcomm", "SM7435P"),
    )
    private val additionalSupportedPlatforms = setOf("parrot")

    val checker = object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean = isSupportedQualcommSoc(
            sdkInt = Build.VERSION.SDK_INT,
            manufacturer = Build.SOC_MANUFACTURER,
            model = Build.SOC_MODEL,
            board = Build.BOARD,
            hardware = Build.HARDWARE,
            liteRtRecognized = NpuCompatibilityChecker.Qualcomm.isDeviceSupported(),
        )
    }

    internal fun isSupportedQualcommSoc(
        sdkInt: Int,
        manufacturer: String,
        model: String,
        board: String = "",
        hardware: String = "",
        liteRtRecognized: Boolean,
    ): Boolean = sdkInt >= 31 && (
        liteRtRecognized ||
            QualcommSoc(manufacturer, model) in additionalSupportedSocs ||
            board.lowercase(Locale.ROOT) in additionalSupportedPlatforms ||
            hardware.lowercase(Locale.ROOT) in additionalSupportedPlatforms
    )

    private data class QualcommSoc(
        val manufacturer: String,
        val model: String,
    )
}
