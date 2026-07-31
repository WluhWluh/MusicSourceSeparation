package com.example.musicsourceseparation.benchmark

import org.json.JSONObject

internal data class QnnIrFileEvidence(
    val path: String,
    val bytes: Long,
)

internal enum class QnnDelegationStatus(val id: String) {
    DELEGATED("delegated"),
    CPU_FALLBACK("cpu_fallback"),
    PROVIDER_UNAVAILABLE("provider_unavailable"),
    INDETERMINATE("indeterminate"),
}

internal data class QnnDelegationAssessment(
    val status: QnnDelegationStatus,
    val verifiedNpuExecution: Boolean,
    val irFileCount: Int,
    val nonEmptyIrFileCount: Int,
    val irBytes: Long,
    val reason: String,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("status", status.id)
        .put("verifiedNpuExecution", verifiedNpuExecution)
        .put("irFileCount", irFileCount)
        .put("nonEmptyIrFileCount", nonEmptyIrFileCount)
        .put("irBytes", irBytes)
        .put("reason", reason)
}

internal object QnnDelegationEvidence {
    fun assess(
        deviceSupported: Boolean?,
        libraryReady: Boolean?,
        irFilesDeclared: Boolean,
        irFiles: List<QnnIrFileEvidence>,
        malformedIrFileCount: Int = 0,
    ): QnnDelegationAssessment {
        val nonEmptyFiles = irFiles.filter { it.path.isNotBlank() && it.bytes > 0L }
        val irFileCount = irFiles.size + malformedIrFileCount
        val irBytes = nonEmptyFiles.sumOf { it.bytes }
        if (nonEmptyFiles.isNotEmpty()) {
            return QnnDelegationAssessment(
                status = QnnDelegationStatus.DELEGATED,
                verifiedNpuExecution = true,
                irFileCount = irFileCount,
                nonEmptyIrFileCount = nonEmptyFiles.size,
                irBytes = irBytes,
                reason = "LiteRT emitted at least one non-empty Qualcomm QNN IR partition.",
            )
        }
        if (deviceSupported == false || libraryReady == false) {
            return QnnDelegationAssessment(
                status = QnnDelegationStatus.PROVIDER_UNAVAILABLE,
                verifiedNpuExecution = false,
                irFileCount = irFileCount,
                nonEmptyIrFileCount = 0,
                irBytes = 0L,
                reason = "The Qualcomm provider did not pass device and library preflight.",
            )
        }
        if (deviceSupported == null || libraryReady == null || !irFilesDeclared || malformedIrFileCount > 0) {
            return QnnDelegationAssessment(
                status = QnnDelegationStatus.INDETERMINATE,
                verifiedNpuExecution = false,
                irFileCount = irFileCount,
                nonEmptyIrFileCount = 0,
                irBytes = 0L,
                reason = "The report does not contain complete, valid QNN delegation evidence.",
            )
        }
        return QnnDelegationAssessment(
            status = QnnDelegationStatus.CPU_FALLBACK,
            verifiedNpuExecution = false,
            irFileCount = irFileCount,
            nonEmptyIrFileCount = 0,
            irBytes = 0L,
            reason = "The provider was ready, but LiteRT emitted no non-empty QNN IR; the completed run used CPU fallback or otherwise failed to verify NPU delegation.",
        )
    }

    fun assess(evidence: JSONObject): QnnDelegationAssessment {
        val irFilesDeclared = evidence.has("irFiles") && !evidence.isNull("irFiles")
        val array = evidence.optJSONArray("irFiles")
        val files = mutableListOf<QnnIrFileEvidence>()
        var malformed = 0
        if (array != null) {
            repeat(array.length()) { index ->
                val file = array.optJSONObject(index)
                if (file == null || !file.has("path") || !file.has("bytes")) {
                    malformed += 1
                } else {
                    files += QnnIrFileEvidence(
                        path = file.optString("path", ""),
                        bytes = file.optLong("bytes", -1L),
                    )
                }
            }
        }
        return assess(
            deviceSupported = evidence.optionalBoolean("deviceSupported"),
            libraryReady = evidence.optionalBoolean("libraryReady"),
            irFilesDeclared = irFilesDeclared,
            irFiles = files,
            malformedIrFileCount = malformed,
        )
    }

    fun annotate(evidence: JSONObject): QnnDelegationAssessment = assess(evidence).also { assessment ->
        evidence
            .put("delegationStatus", assessment.status.id)
            .put("delegationVerified", assessment.verifiedNpuExecution)
            .put("delegationReason", assessment.reason)
            .put("irFileCount", assessment.irFileCount)
            .put("nonEmptyIrFileCount", assessment.nonEmptyIrFileCount)
            .put("irBytes", assessment.irBytes)
    }

    private fun JSONObject.optionalBoolean(name: String): Boolean? =
        if (has(name) && !isNull(name)) getBoolean(name) else null
}
