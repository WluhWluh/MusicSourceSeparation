package com.example.musicsourceseparation.runtime

import android.content.Context
import android.os.Build
import android.os.Process
import android.os.SystemClock
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.ZipFile

internal data class DownloadableLiteRtCoreArtifact(
    val abi: String,
    val bundleFileName: String,
    val bundleBytes: Long,
    val bundleSha256: String,
    val manifestSha256: String,
    val libraryBytes: Long,
    val librarySha256: String,
    val soname: String,
) {
    val downloadUrl: String
        get() = "$RELEASE_BASE_URL/$bundleFileName"

    companion object {
        const val RELEASE_VERSION = "2.1.5-bss.2-exp.2"
        const val RUNTIME_ARTIFACT_VERSION = "2.1.5-bss.2"
        const val LIBRARY_NAME = "libLiteRt.so"
        private const val RELEASE_BASE_URL =
                "https://github.com/WluhWluh/bss-litert-android/releases/download/" +
                "downloadable-runtime-v2.1.5-bss.2-exp.2"

        private val artifacts = listOf(
            DownloadableLiteRtCoreArtifact(
                abi = "arm64-v8a",
                bundleFileName = "litert-cpu-core-2.1.5-bss.2-arm64-v8a.zip",
                bundleBytes = 2_220_465,
                bundleSha256 = "ba566a2b0d3ee95190bced05bf20dff790ef90602af066fc502540c160a0ce49",
                manifestSha256 = "15e04dd49bb377b25bad8f96752dc13e10b84baff508a54cdf4d0bc4b974d7de",
                libraryBytes = 5_328_296,
                librarySha256 = "ae2b996fde27021b070e88b56eebc9626a5261feb72f09791bdac38b2f09abd2",
                soname = "libLiteRt.so",
            ),
            DownloadableLiteRtCoreArtifact(
                abi = "armeabi-v7a",
                bundleFileName = "litert-cpu-core-2.1.5-bss.2-armeabi-v7a.zip",
                bundleBytes = 1_843_957,
                bundleSha256 = "a5c21e9c64030ae9a2a29c6d458e3f7b63018b766981064d570f19082c2c19c1",
                manifestSha256 = "0b765913fa1ca1c531530e2bed17e971a88d3cbf317c8e1c758b60d148bc6cdc",
                libraryBytes = 3_504_124,
                librarySha256 = "836ee7a2321c9453f02658b6774fc4c5951716432b450ba6bc4e9a94fe524e6c",
                soname = "libLiteRt.so",
            ),
            DownloadableLiteRtCoreArtifact(
                abi = "x86_64",
                bundleFileName = "litert-cpu-core-2.1.5-bss.2-x86_64.zip",
                bundleBytes = 2_938_161,
                bundleSha256 = "0425958720617ee00689af1efb1ac2dd03da8481ad0e03c42e0dff7006bc4d85",
                manifestSha256 = "ec4249fda523203e3ef7e85565071dadeac5a742b19b241d1f933ca6a2005148",
                libraryBytes = 7_272_904,
                librarySha256 = "6d5b2f35d536a3b2d38b26d26328cc9c259133ef2aa0413ec554cd7ef84f6604",
                soname = "libLiteRt.so",
            ),
            DownloadableLiteRtCoreArtifact(
                abi = "x86",
                bundleFileName = "litert-cpu-core-2.1.5-bss.2-x86.zip",
                bundleBytes = 2_886_192,
                bundleSha256 = "8ece235a9c1da2478c0ff6d5f7f13908aaca5dec7c8393a36c8058975bd4c975",
                manifestSha256 = "e3e06af11982bf2e02ea1751c889935c4d67c869a57d0eeffa2691dd9bf1679a",
                libraryBytes = 7_482_132,
                librarySha256 = "02b6556ec235926c11eb0c067eb16e459adcddb1568a42eefe0c40f4cc4b59af",
                soname = "LiteRt",
            ),
        )

        fun forProcess(
            is64Bit: Boolean,
            supported64BitAbis: List<String>,
            supported32BitAbis: List<String>,
        ): DownloadableLiteRtCoreArtifact {
            val processAbis = if (is64Bit) supported64BitAbis else supported32BitAbis
            return requireNotNull(processAbis.firstNotNullOfOrNull { abi ->
                artifacts.firstOrNull { artifact -> artifact.abi == abi }
            }) {
                "No downloadable LiteRT core for process ABIs $processAbis"
            }
        }
    }
}

internal object DownloadableLiteRtCore {
    private const val ROOT_DIRECTORY = "downloadable-litert-core"
    private const val MANIFEST_NAME = "manifest.json"
    private const val MAX_MANIFEST_BYTES = 64 * 1024

    @Volatile
    private var loadedReport: String? = null

    @Volatile
    private var lastAttemptReport: String? = null

    @Synchronized
    fun ensureLoaded(context: Context): JSONObject {
        loadedReport?.let { existing ->
            return JSONObject(existing).put("reusedInProcess", true)
        }
        val artifact = DownloadableLiteRtCoreArtifact.forProcess(
            is64Bit = Process.is64Bit(),
            supported64BitAbis = Build.SUPPORTED_64_BIT_ABIS.toList(),
            supported32BitAbis = Build.SUPPORTED_32_BIT_ABIS.toList(),
        )
        val started = SystemClock.elapsedRealtimeNanos()
        return try {
            val installation = ensureInstalled(context.applicationContext, artifact)
            val loader = DownloadableLiteRtNativeLoader.configureAndLoad(installation.library)
            val report = baseReport(artifact)
                .put("status", "loaded")
                .put("downloaded", installation.downloaded)
                .put("libraryPath", installation.library.absolutePath)
                .put("libraryWritable", installation.library.canWrite())
                .put("nativeLoader", loader.toJson())
                .put("totalWallMs", nanosToMs(SystemClock.elapsedRealtimeNanos() - started))
                .put("reusedInProcess", false)
            val serialized = report.toString()
            loadedReport = serialized
            lastAttemptReport = serialized
            JSONObject(serialized)
        } catch (error: Throwable) {
            val report = baseReport(artifact)
                .put("status", "error")
                .put("totalWallMs", nanosToMs(SystemClock.elapsedRealtimeNanos() - started))
                .put("errorType", error::class.java.name)
                .put("message", error.message ?: error::class.java.simpleName)
            lastAttemptReport = report.toString()
            throw IllegalStateException(
                "Downloadable LiteRT core load failed for ${artifact.abi}: ${error.message}",
                error,
            )
        }
    }

    fun lastAttempt(): JSONObject? = lastAttemptReport?.let(::JSONObject)

    private fun ensureInstalled(
        context: Context,
        artifact: DownloadableLiteRtCoreArtifact,
    ): Installation {
        val root = File(context.noBackupFilesDir, ROOT_DIRECTORY).apply {
            require(isDirectory || mkdirs()) { "Could not create runtime root: $absolutePath" }
        }
        val finalDirectory = File(
            root,
            "${DownloadableLiteRtCoreArtifact.RELEASE_VERSION}/${artifact.abi}",
        )
        installedLibrary(finalDirectory, artifact)?.let { library ->
            return Installation(library, downloaded = false)
        }
        if (finalDirectory.exists()) {
            require(finalDirectory.deleteRecursively()) {
                "Could not remove invalid runtime directory: ${finalDirectory.absolutePath}"
            }
        }

        val staging = File(root, ".staging/${UUID.randomUUID()}")
        require(staging.mkdirs()) { "Could not create runtime staging directory" }
        try {
            val bundle = File(staging, artifact.bundleFileName)
            downloadBundle(artifact, bundle)
            val install = File(staging, "install").apply {
                require(mkdir()) { "Could not create staged runtime installation" }
            }
            extractAndVerify(bundle, install, artifact)
            val manifest = File(install, MANIFEST_NAME)
            val library = File(install, DownloadableLiteRtCoreArtifact.LIBRARY_NAME)
            makeReadOnly(manifest, executable = false)
            makeReadOnly(library, executable = true)

            require(finalDirectory.parentFile?.isDirectory == true || finalDirectory.parentFile?.mkdirs() == true) {
                "Could not create final runtime parent directory"
            }
            try {
                Files.move(
                    install.toPath(),
                    finalDirectory.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(install.toPath(), finalDirectory.toPath())
            }
            val installed = requireNotNull(installedLibrary(finalDirectory, artifact)) {
                "Installed runtime failed post-move verification"
            }
            return Installation(installed, downloaded = true)
        } finally {
            staging.deleteRecursively()
        }
    }

    private fun installedLibrary(
        directory: File,
        artifact: DownloadableLiteRtCoreArtifact,
    ): File? {
        val manifest = File(directory, MANIFEST_NAME)
        val library = File(directory, DownloadableLiteRtCoreArtifact.LIBRARY_NAME)
        if (!manifest.isFile || !library.isFile) return null
        return if (
            manifest.length() <= MAX_MANIFEST_BYTES &&
            sha256(manifest) == artifact.manifestSha256 &&
            library.length() == artifact.libraryBytes &&
            sha256(library) == artifact.librarySha256 &&
            validateManifest(manifest.readText(), artifact)
        ) {
            library
        } else {
            null
        }
    }

    private fun downloadBundle(
        artifact: DownloadableLiteRtCoreArtifact,
        destination: File,
    ) {
        val connection = URL(artifact.downloadUrl).openConnection() as HttpURLConnection
        connection.instanceFollowRedirects = true
        connection.connectTimeout = 30_000
        connection.readTimeout = 120_000
        connection.setRequestProperty("User-Agent", "MusicSourceSeparation-LiteRT-Core-Probe/1")
        try {
            require(connection.responseCode in 200..299) {
                "Runtime download returned HTTP ${connection.responseCode}"
            }
            FileOutputStream(destination).use { output ->
                connection.inputStream.buffered().use { input ->
                    val buffer = ByteArray(64 * 1024)
                    var total = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= artifact.bundleBytes) { "Runtime bundle exceeds expected size" }
                        output.write(buffer, 0, count)
                    }
                    require(total == artifact.bundleBytes) {
                        "Runtime bundle size mismatch: $total != ${artifact.bundleBytes}"
                    }
                }
                output.fd.sync()
            }
        } finally {
            connection.disconnect()
        }
        require(sha256(destination) == artifact.bundleSha256) {
            "Runtime bundle SHA-256 mismatch"
        }
    }

    private fun extractAndVerify(
        bundle: File,
        destination: File,
        artifact: DownloadableLiteRtCoreArtifact,
    ) {
        var manifest: ByteArray? = null
        var libraryResult: StreamCopyResult? = null
        val seen = mutableSetOf<String>()
        val libraryFile = File(destination, DownloadableLiteRtCoreArtifact.LIBRARY_NAME)
        ZipFile(bundle).use { archive ->
            val entries = archive.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                if (entry.isDirectory) continue
                require(entry.name in setOf(MANIFEST_NAME, DownloadableLiteRtCoreArtifact.LIBRARY_NAME)) {
                    "Unexpected runtime bundle entry: ${entry.name}"
                }
                require(seen.add(entry.name)) { "Duplicate runtime bundle entry: ${entry.name}" }
                archive.getInputStream(entry).use { input ->
                    if (entry.name == MANIFEST_NAME) {
                        manifest = readBounded(input, MAX_MANIFEST_BYTES.toLong())
                    } else {
                        FileOutputStream(libraryFile).use { output ->
                            libraryResult = copyBoundedAndHash(
                                input = input,
                                output = output,
                                maximum = artifact.libraryBytes,
                            )
                            output.fd.sync()
                        }
                    }
                }
            }
            require(seen == setOf(MANIFEST_NAME, DownloadableLiteRtCoreArtifact.LIBRARY_NAME)) {
                "Runtime bundle file set is incomplete"
            }
            val verifiedManifest = requireNotNull(manifest)
            val verifiedLibrary = requireNotNull(libraryResult)
            require(sha256(verifiedManifest) == artifact.manifestSha256) {
                "Manifest SHA-256 mismatch"
            }
            require(validateManifest(verifiedManifest.toString(Charsets.UTF_8), artifact)) {
                "Runtime manifest contract mismatch"
            }
            require(verifiedLibrary.byteSize == artifact.libraryBytes) { "Library size mismatch" }
            require(verifiedLibrary.sha256 == artifact.librarySha256) {
                "Library SHA-256 mismatch"
            }
            writeSynced(File(destination, MANIFEST_NAME), verifiedManifest)
        }
    }

    private fun validateManifest(
        text: String,
        artifact: DownloadableLiteRtCoreArtifact,
    ): Boolean = runCatching {
        val manifest = JSONObject(text)
        val files = manifest.getJSONArray("files")
        val library = files.getJSONObject(0)
        val elf = library.getJSONObject("elf")
        manifest.getInt("schemaVersion") == 1 &&
            manifest.getString("contractSchemaVersion") == "bss-litert-downloadable-runtime-v2" &&
            manifest.getString("releaseVersion") == DownloadableLiteRtCoreArtifact.RELEASE_VERSION &&
            manifest.getString("runtimeArtifactVersion") ==
                DownloadableLiteRtCoreArtifact.RUNTIME_ARTIFACT_VERSION &&
            manifest.getString("component") == "cpu-core" &&
            manifest.getString("abi") == artifact.abi &&
            files.length() == 1 &&
            library.getString("path") == DownloadableLiteRtCoreArtifact.LIBRARY_NAME &&
            library.getLong("byteSize") == artifact.libraryBytes &&
            library.getString("sha256") == artifact.librarySha256 &&
            elf.getString("soname") == artifact.soname
    }.getOrDefault(false)

    private fun readBounded(input: java.io.InputStream, maximum: Long): ByteArray {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(64 * 1024)
        var total = 0L
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            total += count
            require(total <= maximum) { "Runtime archive entry exceeds expected size" }
            output.write(buffer, 0, count)
        }
        return output.toByteArray()
    }

    private fun writeSynced(file: File, data: ByteArray) {
        FileOutputStream(file).use { output ->
            output.write(data)
            output.fd.sync()
        }
    }

    private fun makeReadOnly(file: File, executable: Boolean) {
        require(file.setReadable(true, true)) { "Could not make ${file.name} readable" }
        require(file.setWritable(false, false)) { "Could not make ${file.name} read-only" }
        if (executable) {
            require(file.setExecutable(true, true)) { "Could not make ${file.name} executable" }
        }
    }

    private fun baseReport(artifact: DownloadableLiteRtCoreArtifact): JSONObject = JSONObject()
        .put("schemaVersion", 1)
        .put("releaseVersion", DownloadableLiteRtCoreArtifact.RELEASE_VERSION)
        .put("runtimeArtifactVersion", DownloadableLiteRtCoreArtifact.RUNTIME_ARTIFACT_VERSION)
        .put("abi", artifact.abi)
        .put("processIs64Bit", Process.is64Bit())
        .put("supportedAbis", Build.SUPPORTED_ABIS.joinToString(","))
        .put("bundleFileName", artifact.bundleFileName)
        .put("bundleBytes", artifact.bundleBytes)
        .put("bundleSha256", artifact.bundleSha256)
        .put("bundleUrl", artifact.downloadUrl)
        .put("manifestSha256", artifact.manifestSha256)
        .put("libraryBytes", artifact.libraryBytes)
        .put("librarySha256", artifact.librarySha256)
        .put("soname", artifact.soname)

    private fun sha256(file: File): String = file.inputStream().buffered().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(1024 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private fun sha256(data: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(data)
        .joinToString("") { byte -> "%02x".format(byte) }

    private fun nanosToMs(nanoseconds: Long): Double = nanoseconds / 1_000_000.0

    private data class Installation(
        val library: File,
        val downloaded: Boolean,
    )
}

internal data class StreamCopyResult(
    val byteSize: Long,
    val sha256: String,
)

internal fun copyBoundedAndHash(
    input: InputStream,
    output: OutputStream,
    maximum: Long,
): StreamCopyResult {
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(64 * 1024)
    var total = 0L
    while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        total += count
        require(total <= maximum) { "Runtime archive entry exceeds expected size" }
        digest.update(buffer, 0, count)
        output.write(buffer, 0, count)
    }
    return StreamCopyResult(
        byteSize = total,
        sha256 = digest.digest().joinToString("") { byte -> "%02x".format(byte) },
    )
}
