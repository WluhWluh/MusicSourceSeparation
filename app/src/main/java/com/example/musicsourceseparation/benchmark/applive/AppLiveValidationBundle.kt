package com.example.musicsourceseparation.benchmark.applive

import android.content.Context
import org.json.JSONObject
import java.net.URI

internal data class AppLiveFixture(
    val id: String,
    val assetPath: String,
    val targetDirectory: String,
    val targetName: String,
    val bytes: Long,
    val sha256: String,
)

internal data class AppLiveRelayConfig(
    val baseUrl: String,
    val campaign: String,
    val uploadToken: String,
)

internal data class AppLiveValidationBundle(
    val bundleId: String,
    val diagnosticContractVersion: Int,
    val sourceCommit: String,
    val sourceDirty: Boolean,
    val manifestText: String,
    val fixtures: Map<String, AppLiveFixture>,
    val relay: AppLiveRelayConfig,
) {
    fun requireFixture(id: String): AppLiveFixture = requireNotNull(fixtures[id]) {
        "App Live fixture is missing from the bundle: $id"
    }

    companion object {
        private const val MANIFEST_ASSET = "app-live/manifest.json"
        private const val CONFIG_ASSET = "app-live/config.json"
        private val safeName = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
        private val sha256 = Regex("[0-9a-f]{64}")
        private val allowedTargetDirectories = setOf(
            "models",
            "inputs",
            "reference",
            "audio-input",
            "evidence",
        )

        fun load(context: Context): AppLiveValidationBundle {
            val manifestText = context.assets.open(MANIFEST_ASSET).bufferedReader().use { it.readText() }
            val configText = context.assets.open(CONFIG_ASSET).bufferedReader().use { it.readText() }
            val manifest = JSONObject(manifestText)
            val config = JSONObject(configText)
            require(manifest.getInt("schemaVersion") == 1) { "Unsupported App Live manifest schema" }
            require(config.getInt("schemaVersion") == 1) { "Unsupported App Live config schema" }

            val bundleId = manifest.getString("bundleId").lowercase()
            require(bundleId.matches(sha256)) { "Invalid App Live bundle ID" }
            val diagnosticContractVersion = manifest.optInt("diagnosticContractVersion", 1)
            require(diagnosticContractVersion in 1..2) {
                "Unsupported App Live diagnostic contract: $diagnosticContractVersion"
            }
            val source = manifest.getJSONObject("source")
            val fileArray = manifest.getJSONArray("files")
            val fixtures = buildMap {
                repeat(fileArray.length()) { index ->
                    val file = fileArray.getJSONObject(index)
                    val id = file.getString("id")
                    val assetPath = file.getString("asset")
                    val targetDirectory = file.getString("targetDirectory")
                    val targetName = file.getString("targetName")
                    val expectedSha256 = file.getString("sha256").lowercase()
                    require(id.matches(safeName)) { "Invalid fixture ID: $id" }
                    require(assetPath.matches(Regex("app-live/[A-Za-z0-9][A-Za-z0-9._-]{0,95}"))) {
                        "Invalid fixture asset path: $assetPath"
                    }
                    require(targetDirectory in allowedTargetDirectories) {
                        "Invalid fixture target directory: $targetDirectory"
                    }
                    require(targetName.matches(safeName)) { "Invalid fixture target name: $targetName" }
                    require(expectedSha256.matches(sha256)) { "Invalid fixture SHA-256: $id" }
                    require(!containsKey(id)) { "Duplicate fixture ID: $id" }
                    put(
                        id,
                        AppLiveFixture(
                            id = id,
                            assetPath = assetPath,
                            targetDirectory = targetDirectory,
                            targetName = targetName,
                            bytes = file.getLong("bytes"),
                            sha256 = expectedSha256,
                        ),
                    )
                }
            }
            val baseUrl = config.getString("baseUrl").trimEnd('/')
            val uri = URI(baseUrl)
            require(uri.scheme == "https" && !uri.host.isNullOrBlank()) { "Relay URL must use HTTPS" }
            val campaign = config.getString("campaign")
            require(campaign.matches(safeName)) { "Invalid relay campaign" }
            val uploadToken = config.getString("uploadToken")
            require(uploadToken.length >= 32) { "Relay upload token is missing" }
            REQUIRED_FIXTURES.forEach { id -> require(id in fixtures) { "Required fixture is missing: $id" } }
            return AppLiveValidationBundle(
                bundleId = bundleId,
                diagnosticContractVersion = diagnosticContractVersion,
                sourceCommit = source.getString("commit"),
                sourceDirty = source.getBoolean("dirty"),
                manifestText = manifestText,
                fixtures = fixtures,
                relay = AppLiveRelayConfig(baseUrl, campaign, uploadToken),
            )
        }

        fun loadCatching(context: Context): Result<AppLiveValidationBundle> = runCatching { load(context) }

        private val REQUIRED_FIXTURES = setOf(
            "model",
            "windowInput",
            "windowReference",
            "quickAudio",
            "fullAudio",
            "runtimeManifest",
        )
    }
}
