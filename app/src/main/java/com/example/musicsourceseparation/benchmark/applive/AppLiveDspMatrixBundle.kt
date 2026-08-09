package com.example.musicsourceseparation.benchmark.applive

import android.content.Context
import org.json.JSONObject
import java.net.URI

internal data class AppLiveDspShape(
    val id: String,
    val nFft: Int,
    val hopLength: Int,
    val dimF: Int,
    val dimTPower: Int,
)

internal data class AppLiveDspMatrixBundle(
    val bundleId: String,
    val contractVersion: Int,
    val sourceCommit: String,
    val sourceDirty: Boolean,
    val manifestText: String,
    val workerCount: Int,
    val warmups: Int,
    val measuredRuns: Int,
    val minimumSnrDb: Double,
    val maximumAbsoluteError: Double,
    val autoStart: Boolean,
    val shapes: List<AppLiveDspShape>,
    val relay: AppLiveRelayConfig,
) {
    companion object {
        private const val MANIFEST_ASSET = "app-live-dsp/manifest.json"
        private const val CONFIG_ASSET = "app-live-dsp/config.json"
        private val safeName = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
        private val sha256 = Regex("[0-9a-f]{64}")
        private val expectedShapes = mapOf(
            "uvr_mdxnet_3_9662" to intArrayOf(6_144, 1_024, 2_048, 8),
            "kim_inst" to intArrayOf(7_680, 1_024, 3_072, 8),
            "uvr_mdxnet_inst_hq_4" to intArrayOf(5_120, 1_024, 2_560, 8),
        )

        fun load(context: Context): AppLiveDspMatrixBundle {
            val manifestText = context.assets.open(MANIFEST_ASSET).bufferedReader().use { it.readText() }
            val configText = context.assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() }
            val manifest = JSONObject(manifestText)
            val config = JSONObject(configText)
            require(manifest.getInt("schemaVersion") == 1) { "Unsupported DSP matrix manifest schema" }
            require(config.getInt("schemaVersion") == 1) { "Unsupported DSP matrix config schema" }
            val bundleId = manifest.getString("bundleId").lowercase()
            require(bundleId.matches(sha256)) { "Invalid DSP matrix bundle ID" }
            val contractVersion = manifest.getInt("contractVersion")
            require(contractVersion == 2) { "Unsupported DSP matrix contract: $contractVersion" }
            val source = manifest.getJSONObject("source")
            val matrix = manifest.getJSONObject("matrix")
            val profiles = matrix.getJSONArray("profiles")
            require(List(profiles.length()) { profiles.getString(it) } == EXPECTED_PROFILES) {
                "Unexpected DSP matrix profiles"
            }
            val shapesJson = matrix.getJSONArray("shapes")
            val shapes = buildList {
                repeat(shapesJson.length()) { index ->
                    val value = shapesJson.getJSONObject(index)
                    val shape = AppLiveDspShape(
                        id = value.getString("id"),
                        nFft = value.getInt("nFft"),
                        hopLength = value.getInt("hopLength"),
                        dimF = value.getInt("dimF"),
                        dimTPower = value.getInt("dimTPower"),
                    )
                    val expected = expectedShapes[shape.id]
                    require(expected != null && expected.contentEquals(
                        intArrayOf(shape.nFft, shape.hopLength, shape.dimF, shape.dimTPower),
                    )) { "Unexpected DSP shape contract: ${shape.id}" }
                    add(shape)
                }
            }
            require(shapes.map { it.id }.toSet() == expectedShapes.keys) {
                "DSP matrix must contain the three frozen sentinel shapes"
            }
            val baseUrl = config.getString("baseUrl").trimEnd('/')
            val uri = URI(baseUrl)
            require(uri.scheme == "https" && !uri.host.isNullOrBlank()) { "Relay URL must use HTTPS" }
            val campaign = config.getString("campaign")
            require(campaign.matches(safeName)) { "Invalid relay campaign" }
            val uploadToken = config.getString("uploadToken")
            require(uploadToken.length >= 32) { "Relay upload token is missing" }
            val minimumSnrDb = matrix.getDouble("minimumSnrDb")
            val maximumAbsoluteError = matrix.getDouble("maximumAbsoluteError")
            require(minimumSnrDb >= 80.0) { "DSP matrix SNR gate is too weak" }
            require(maximumAbsoluteError in 0.0..0.001) { "DSP matrix absolute-error gate is too weak" }
            return AppLiveDspMatrixBundle(
                bundleId = bundleId,
                contractVersion = contractVersion,
                sourceCommit = source.getString("commit"),
                sourceDirty = source.getBoolean("dirty"),
                manifestText = manifestText,
                workerCount = matrix.getInt("workerCount").also { require(it in 1..8) },
                warmups = matrix.getInt("warmups").also { require(it in 1..5) },
                measuredRuns = matrix.getInt("measuredRuns").also { require(it in 2..50 && it % 2 == 0) },
                minimumSnrDb = minimumSnrDb,
                maximumAbsoluteError = maximumAbsoluteError,
                autoStart = config.optBoolean("autoStart", true),
                shapes = shapes,
                relay = AppLiveRelayConfig(baseUrl, campaign, uploadToken),
            )
        }

        fun loadCatching(context: Context): Result<AppLiveDspMatrixBundle> = runCatching { load(context) }

        private val EXPECTED_PROFILES = listOf(
            "kotlin-jtransforms",
            "native-full",
            "native-packed",
        )
    }
}
