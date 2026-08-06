import java.security.MessageDigest

plugins {
    id("com.android.application")
}

fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
}

fun buildConfigString(value: String): String =
    "\"${value.replace("\\", "\\\\").replace("\"", "\\\"")}\""

val liteRtAar = providers.gradleProperty("liteRtAar").orNull
val qnnHtpVersions = listOf(69, 73, 75, 79)
val sourceRevision = providers.gradleProperty("benchmarkSourceRevision").orElse("unknown").get()
val sourceDirty = providers.gradleProperty("benchmarkSourceDirty").orElse("unknown").get()
val runtimeId = providers.gradleProperty("benchmarkRuntimeId").orElse(
    if (liteRtAar == null) "com.google.ai.edge.litert:litert:2.1.5" else "local-litert-aar",
).get()
val runtimeArtifactSha256 = providers.gradleProperty("benchmarkRuntimeArtifactSha256").orNull
    ?: liteRtAar?.let { sha256(file(it)) }
    ?: "maven-unresolved"

android {
    namespace = "com.example.musicsourceseparation"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.example.musicsourceseparation"
        minSdk = 26
        targetSdk = 37
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("String", "BENCHMARK_SOURCE_REVISION", buildConfigString(sourceRevision))
        buildConfigField("String", "BENCHMARK_SOURCE_DIRTY", buildConfigString(sourceDirty))
        buildConfigField("String", "BENCHMARK_RUNTIME_ID", buildConfigString(runtimeId))
        buildConfigField("String", "BENCHMARK_RUNTIME_VERSION", buildConfigString("2.1.5"))
        buildConfigField(
            "String",
            "BENCHMARK_RUNTIME_ARTIFACT_SHA256",
            buildConfigString(runtimeArtifactSha256),
        )
        buildConfigField("String", "BENCHMARK_ACCELERATOR_BUNDLE_SHA256", buildConfigString("none"))
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    flavorDimensions += "runtime"
    productFlavors {
        create("standard") {
            dimension = "runtime"
        }
        qnnHtpVersions.forEach { htpVersion ->
            create("qnnV$htpVersion") {
                dimension = "runtime"
                minSdk = 31
                ndk {
                    abiFilters += "arm64-v8a"
                }
                val runtimeManifest = file("../.tmp/litert-qnn-v$htpVersion-runtime/runtime-manifest.json")
                val manifestSha256 = if (runtimeManifest.isFile) sha256(runtimeManifest) else "missing"
                buildConfigField(
                    "String",
                    "BENCHMARK_ACCELERATOR_BUNDLE_SHA256",
                    buildConfigString(manifestSha256),
                )
            }
        }
    }

    sourceSets {
        getByName("main") {
            assets.directories.add("../models/uvr-mdx")
        }
        qnnHtpVersions.forEach { htpVersion ->
            getByName("qnnV$htpVersion") {
                jniLibs.directories.add("../.tmp/litert-qnn-v$htpVersion-runtime/jni")
                assets.directories.add("../.tmp/litert-qnn-v$htpVersion-runtime/licenses")
            }
        }
    }

    packaging {
        jniLibs {
            // BuiltinNpuAcceleratorProvider discovers plugins through nativeLibraryDir.
            useLegacyPackaging = true
            keepDebugSymbols += setOf(
                "**/libLiteRt*Qualcomm.so",
                "**/libQnn*.so",
            )
        }
    }
}

dependencies {
    // 2.1.6 currently publishes litert and litert-api with the same AAR namespace,
    // which AGP 9.2 rejects. 2.1.5 contains the same CompiledModel GPU API in one AAR.
    if (liteRtAar != null) {
        implementation(files(liteRtAar))
    } else {
        implementation("com.google.ai.edge.litert:litert:2.1.5")
    }
    // Historical x86 benchmark artifact. Canonical builds are published by
    // https://github.com/WluhWluh/bss-litert-android.
    implementation(files("libs/litert-2.1.5-x86.aar"))
    implementation("com.github.wendykierp:JTransforms:3.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}
