package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.model.HtdemucsDsp
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HtdemucsCanonicalE2eInstrumentedTest {
    @Test
    fun runCanonicalCpuE2e() {
        val arguments = InstrumentationRegistry.getArguments()
        val audioFile = requireNotNull(
            arguments.getString(HtdemucsCanonicalE2eBenchmark.ARG_AUDIO_FILE),
        ) {
            "Pass -e ${HtdemucsCanonicalE2eBenchmark.ARG_AUDIO_FILE} <audio-base-name>."
        }
        val requestedDurationSeconds = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_DURATION_SECONDS)
            ?.toIntOrNull()
        val frameLimit = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_FRAME_LIMIT)
            ?.toIntOrNull()
        val durationSeconds = requestedDurationSeconds ?: if (frameLimit == null) 30 else null
        val threads = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_THREADS)
            ?.toIntOrNull()
            ?: 4
        val dspMode = HtdemucsCanonicalE2eBenchmark.DspMode.fromWireValue(
            arguments.getString(HtdemucsCanonicalE2eBenchmark.ARG_DSP_MODE)
                ?: "kotlin-jtransforms",
        )
        val istftMode = HtdemucsDsp.IstftMode.fromWireValue(
            arguments.getString(HtdemucsCanonicalE2eBenchmark.ARG_ISTFT_MODE) ?: "serial",
        )
        val istftWorkers = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_ISTFT_WORKERS)
            ?.toIntOrNull()
            ?: 1
        val validateIstftFloatParity = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_VALIDATE_ISTFT_FLOAT_PARITY)
            ?.toBooleanStrictOrNull()
            ?: false
        val coreWarmupRuns = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_CORE_WARMUP_RUNS)
            ?.toIntOrNull()
            ?: 0
        val coreMeasuredRuns = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_CORE_MEASURED_RUNS)
            ?.toIntOrNull()
            ?: 0
        val postprocessMode = HtdemucsCanonicalE2eBenchmark.PostprocessMode.fromWireValue(
            arguments.getString(HtdemucsCanonicalE2eBenchmark.ARG_POSTPROCESS_MODE) ?: "legacy",
        )
        val cancelAfterWindows = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_CANCEL_AFTER_WINDOWS)
            ?.toIntOrNull()
            ?: 0
        val resumeAfterCancel = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_RESUME_AFTER_CANCEL)
            ?.toBooleanStrictOrNull()
            ?: false
        val exportCanonicalInput = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_EXPORT_CANONICAL_INPUT)
            ?.toBooleanStrictOrNull()
            ?: false
        val runId = arguments.getString(HtdemucsCanonicalE2eBenchmark.ARG_RUN_ID)
            ?: "run-${System.currentTimeMillis()}"
        val modelVariant = arguments
            .getString(HtdemucsCanonicalE2eBenchmark.ARG_MODEL_VARIANT)
            ?: HtdemucsCanonicalE2eBenchmark.MODEL_VARIANT_OFFICIAL
        val context = ApplicationProvider.getApplicationContext<Context>()
        val result = HtdemucsCanonicalE2eBenchmark(context).run(
            HtdemucsCanonicalE2eBenchmark.Config(
                audioFileName = audioFile,
                durationSeconds = durationSeconds,
                frameLimit = frameLimit,
                threads = threads,
                dspMode = dspMode,
                istftMode = istftMode,
                istftWorkers = istftWorkers,
                validateIstftFloatParity = validateIstftFloatParity,
                coreWarmupRuns = coreWarmupRuns,
                coreMeasuredRuns = coreMeasuredRuns,
                postprocessMode = postprocessMode,
                runId = runId,
                modelVariant = modelVariant,
                expectedAudioSha256 = arguments.getString(
                    HtdemucsCanonicalE2eBenchmark.ARG_AUDIO_SHA256,
                ),
                cancelAfterWindows = cancelAfterWindows,
                resumeAfterCancel = resumeAfterCancel,
                exportCanonicalInput = exportCanonicalInput,
            ),
        )
        val expectedStatus = if (cancelAfterWindows > 0 && !resumeAfterCancel) {
            "cancelled"
        } else {
            "complete"
        }
        check(result.report.getString("status") == expectedStatus)
        Log.i(LOG_TAG, "Canonical HTDemucs E2E report: ${result.reportFile.absolutePath}")
    }

    private companion object {
        const val LOG_TAG = "MSS-Htdemucs-E2E"
    }
}
