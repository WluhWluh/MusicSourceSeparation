package com.example.musicsourceseparation.benchmark.applive

import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest

internal object AppLiveDspMatrixSummary {
    const val SCHEMA_VERSION = 1

    val csvFields = listOf(
        "run_id", "origin", "status", "bundle_id", "contract_version",
        "source_commit", "source_dirty", "captured_at", "manufacturer", "model",
        "product", "device", "soc_manufacturer", "soc_model", "android_release", "sdk",
        "abi", "total_memory_bytes", "fingerprint_sha256", "apk_sha256", "native_dsp_sha256",
        "pocketfft_revision", "worker_count", "warmups", "measured_runs", "matrix_elapsed_ms",
        "thermal_start", "thermal_end", "battery_temp_start_deci_c", "battery_temp_end_deci_c",
        "pss_start_kb", "pss_end_kb", "native_heap_start_bytes", "native_heap_end_bytes",
        "shape_id", "n_fft", "hop_length", "dim_f", "dim_t", "chunk_size",
        "tensor_elements", "profile", "sample_count", "stft_mean_ms", "stft_median_ms",
        "stft_p95_ms", "istft_mean_ms", "istft_median_ms", "istft_p95_ms",
        "combined_mean_ms", "combined_median_ms", "combined_p95_ms", "process_cpu_median_ms",
        "stft_finite", "stft_snr_db", "stft_max_abs", "istft_finite", "istft_snr_db",
        "istft_max_abs", "bit_exact", "qualified", "speedup_vs_kotlin",
        "speedup_vs_native_full", "native_winner", "shape_pss_start_kb", "shape_pss_end_kb",
        "shape_native_heap_start_bytes", "shape_native_heap_end_bytes", "shape_thermal_start",
        "shape_thermal_end", "shape_gc_count", "shape_blocking_gc_count",
        "shape_allocated_bytes", "shape_freed_bytes",
    )

    fun create(identity: JSONObject, report: JSONObject): JSONObject {
        require(report.getString("status") == "complete")
        require(report.getString("runId") == identity.getString("runId"))
        require(report.getString("bundleId") == identity.getString("bundleId"))
        val device = identity.getJSONObject("device")
        val application = identity.getJSONObject("application")
        val nativeDsp = identity.getJSONObject("nativeDsp")
        val apkSha = application.getJSONArray("apkFiles").getJSONObject(0).getString("sha256")
        val nativeLibraries = application.getJSONArray("nativeLibraries")
        val nativeDspSha = (0 until nativeLibraries.length())
            .map { nativeLibraries.getJSONObject(it) }
            .first { it.getString("name") == "libmss_mdx_dsp.so" }
            .getString("sha256")
        val abis = device.getJSONArray("abis")
        val common = linkedMapOf<String, Any?>(
            "run_id" to report.getString("runId"),
            "origin" to report.getString("origin"),
            "status" to report.getString("status"),
            "bundle_id" to report.getString("bundleId"),
            "contract_version" to report.getInt("contractVersion"),
            "source_commit" to identity.getString("sourceCommit"),
            "source_dirty" to identity.getBoolean("sourceDirty"),
            "captured_at" to identity.getString("capturedAt"),
            "manufacturer" to device.getString("manufacturer"),
            "model" to device.getString("model"),
            "product" to device.getString("product"),
            "device" to device.getString("device"),
            "soc_manufacturer" to device.getString("socManufacturer"),
            "soc_model" to device.getString("socModel"),
            "android_release" to device.getString("androidRelease"),
            "sdk" to device.getInt("sdk"),
            "abi" to if (abis.length() > 0) abis.getString(0) else "",
            "total_memory_bytes" to device.getLong("totalMemoryBytes"),
            "fingerprint_sha256" to sha256(device.getString("fingerprint")),
            "apk_sha256" to apkSha,
            "native_dsp_sha256" to nativeDspSha,
            "pocketfft_revision" to nativeDsp.getString("pocketfftRevision"),
            // Overwritten per shape-run while retaining the frozen CSV field order.
            "worker_count" to 0,
            "warmups" to report.getInt("warmups"),
            "measured_runs" to report.getInt("measuredRunsPerProfile"),
            "matrix_elapsed_ms" to report.getDouble("elapsedMs"),
            "thermal_start" to report.getJSONObject("deviceStart").getInt("thermalStatus"),
            "thermal_end" to report.getJSONObject("deviceEnd").getInt("thermalStatus"),
            "battery_temp_start_deci_c" to report.getJSONObject("deviceStart")
                .getInt("batteryTemperatureDeciC"),
            "battery_temp_end_deci_c" to report.getJSONObject("deviceEnd")
                .getInt("batteryTemperatureDeciC"),
            "pss_start_kb" to report.getJSONObject("processStart").getInt("totalPssKb"),
            "pss_end_kb" to report.getJSONObject("processEnd").getInt("totalPssKb"),
            "native_heap_start_bytes" to report.getJSONObject("processStart")
                .getLong("nativeHeapAllocatedBytes"),
            "native_heap_end_bytes" to report.getJSONObject("processEnd")
                .getLong("nativeHeapAllocatedBytes"),
        )
        val rows = JSONArray()
        val profileIds = report.getJSONArray("profiles").let { values ->
            List(values.length()) { values.getString(it) }
        }
        require(profileIds.firstOrNull() == PROFILE_KOTLIN && PROFILE_NATIVE_PACKED in profileIds)
        val nativeWinners = mutableListOf<String>()
        val shapes = report.getJSONArray("shapes")
        repeat(shapes.length()) { shapeIndex ->
            val shape = shapes.getJSONObject(shapeIndex)
            val config = shape.getJSONObject("config")
            val profiles = shape.getJSONObject("profiles")
            val kotlinMedian = profiles.getJSONObject(PROFILE_KOTLIN)
                .getJSONObject("combined").getDouble("medianMs")
            val packedMedian = profiles.getJSONObject(PROFILE_NATIVE_PACKED)
                .getJSONObject("combined").getDouble("medianMs")
            val fullMedian = profiles.optJSONObject(PROFILE_NATIVE_FULL)
                ?.getJSONObject("combined")?.getDouble("medianMs")
            val nativeWinner = fullMedian?.let {
                if (packedMedian < it) PROFILE_NATIVE_PACKED else PROFILE_NATIVE_FULL
            } ?: "not-measured"
            if (fullMedian != null) nativeWinners += nativeWinner
            val runtime = shape.getJSONObject("runtimeStatsDelta")
            profileIds.forEach { profileId ->
                val profile = profiles.getJSONObject(profileId)
                val stft = profile.getJSONObject("stft")
                val iStft = profile.getJSONObject("iStft")
                val combined = profile.getJSONObject("combined")
                val parity = profile.getJSONObject("parity")
                val stftParity = parity.getJSONObject("stft")
                val iStftParity = parity.getJSONObject("iStft")
                val sampleCount = combined.getInt("count")
                val qualified = sampleCount == report.getInt("measuredRunsPerProfile") &&
                    parityPass(stftParity, report) && parityPass(iStftParity, report)
                val row = JSONObject()
                common.forEach { (key, value) -> row.put(key, value ?: JSONObject.NULL) }
                row.put("worker_count", shape.getInt("workerCount"))
                    .put("shape_id", shape.getString("id"))
                    .put("n_fft", config.getInt("nFft"))
                    .put("hop_length", config.getInt("hopLength"))
                    .put("dim_f", config.getInt("dimF"))
                    .put("dim_t", config.getInt("dimT"))
                    .put("chunk_size", config.getInt("chunkSize"))
                    .put("tensor_elements", config.getInt("tensorElements"))
                    .put("profile", profileId)
                    .put("sample_count", sampleCount)
                    .put("stft_mean_ms", stft.getDouble("meanMs"))
                    .put("stft_median_ms", stft.getDouble("medianMs"))
                    .put("stft_p95_ms", stft.getDouble("p95Ms"))
                    .put("istft_mean_ms", iStft.getDouble("meanMs"))
                    .put("istft_median_ms", iStft.getDouble("medianMs"))
                    .put("istft_p95_ms", iStft.getDouble("p95Ms"))
                    .put("combined_mean_ms", combined.getDouble("meanMs"))
                    .put("combined_median_ms", combined.getDouble("medianMs"))
                    .put("combined_p95_ms", combined.getDouble("p95Ms"))
                    .put("process_cpu_median_ms", profile.getJSONObject("processCpu").getDouble("medianMs"))
                    .put("stft_finite", stftParity.getBoolean("finite"))
                    .put("stft_snr_db", normalizedSnr(stftParity))
                    .put("stft_max_abs", stftParity.getDouble("maxAbs"))
                    .put("istft_finite", iStftParity.getBoolean("finite"))
                    .put("istft_snr_db", normalizedSnr(iStftParity))
                    .put("istft_max_abs", iStftParity.getDouble("maxAbs"))
                    .put("bit_exact", stftParity.getBoolean("bitExact") && iStftParity.getBoolean("bitExact"))
                    .put("qualified", qualified)
                    .put("speedup_vs_kotlin", kotlinMedian / combined.getDouble("medianMs"))
                    .put(
                        "speedup_vs_native_full",
                        fullMedian?.div(combined.getDouble("medianMs")) ?: JSONObject.NULL,
                    )
                    .put("native_winner", nativeWinner)
                    .put("shape_pss_start_kb", shape.getJSONObject("processStart").getInt("totalPssKb"))
                    .put("shape_pss_end_kb", shape.getJSONObject("processEnd").getInt("totalPssKb"))
                    .put(
                        "shape_native_heap_start_bytes",
                        shape.getJSONObject("processStart").getLong("nativeHeapAllocatedBytes"),
                    )
                    .put(
                        "shape_native_heap_end_bytes",
                        shape.getJSONObject("processEnd").getLong("nativeHeapAllocatedBytes"),
                    )
                    .put("shape_thermal_start", shape.getJSONObject("deviceStart").getInt("thermalStatus"))
                    .put("shape_thermal_end", shape.getJSONObject("deviceEnd").getInt("thermalStatus"))
                    .put("shape_gc_count", runtime.getLong("art.gc.gc-count"))
                    .put("shape_blocking_gc_count", runtime.getLong("art.gc.blocking-gc-count"))
                    .put("shape_allocated_bytes", runtime.getLong("art.gc.bytes-allocated"))
                    .put("shape_freed_bytes", runtime.getLong("art.gc.bytes-freed"))
                require(row.length() == csvFields.size) { "Unexpected DSP summary row schema" }
                rows.put(row)
            }
        }
        val uniformWinner = nativeWinners.distinct().singleOrNull()
        val qualifiedRows = (0 until rows.length()).count { rows.getJSONObject(it).getBoolean("qualified") }
        return JSONObject()
            .put("schemaVersion", SCHEMA_VERSION)
            .put("status", report.getString("status"))
            .put("runId", report.getString("runId"))
            .put("bundleId", report.getString("bundleId"))
            .put("contractVersion", report.getInt("contractVersion"))
            .put("sourceCommit", identity.getString("sourceCommit"))
            .put("sourceDirty", identity.getBoolean("sourceDirty"))
            .put("rowCount", rows.length())
            .put("qualifiedRowCount", qualifiedRows)
            .put("allRowsQualified", qualifiedRows == rows.length())
            .put("uniformNativeWinner", uniformWinner ?: if (nativeWinners.isEmpty()) "not-measured" else "mixed")
            .put("nativeWinnerCounts", JSONObject(nativeWinners.groupingBy { it }.eachCount()))
            .put("csvFields", JSONArray(csvFields))
            .put("rows", rows)
    }

    fun toCsv(summary: JSONObject): String {
        val fields = summary.getJSONArray("csvFields")
        require(List(fields.length()) { fields.getString(it) } == csvFields)
        val lines = mutableListOf(csvFields.joinToString(","))
        val rows = summary.getJSONArray("rows")
        repeat(rows.length()) { index ->
            val row = rows.getJSONObject(index)
            lines += csvFields.joinToString(",") { field -> csvEscape(row.opt(field)) }
        }
        return lines.joinToString("\n", postfix = "\n")
    }

    private fun parityPass(parity: JSONObject, report: JSONObject): Boolean =
        parity.getBoolean("finite") &&
            snrValue(parity) >= report.getDouble("minimumSnrDb") &&
            parity.getDouble("maxAbs") <= report.getDouble("maximumAbsoluteError")

    private fun normalizedSnr(parity: JSONObject): Any {
        val value = parity.get("snrDb")
        return if (value is Number) value.toDouble() else "Infinity"
    }

    private fun snrValue(parity: JSONObject): Double {
        val value = parity.get("snrDb")
        return if (value is Number) value.toDouble() else Double.POSITIVE_INFINITY
    }

    private fun csvEscape(value: Any?): String {
        if (value == null || value == JSONObject.NULL) return ""
        val text = value.toString()
        return if (text.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"${text.replace("\"", "\"\"")}\""
        } else {
            text
        }
    }

    private fun sha256(value: String): String = MessageDigest.getInstance("SHA-256")
        .digest(value.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it) }

    private const val PROFILE_KOTLIN = "kotlin-jtransforms"
    private const val PROFILE_NATIVE_FULL = "native-full"
    private const val PROFILE_NATIVE_PACKED = "native-packed"
}
