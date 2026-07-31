import java.security.MessageDigest

plugins {
    id("com.android.application")
}

val liteRtApiAar = providers.gradleProperty("liteRtApiAar").orNull?.let(::file)
val expectedLiteRtApiAarSha256 =
    "2cdac3840bd664109c151da7737811f6c1e8004ab140c4b369f99b339623f0de"
val downloadableCoreTaskRequested = gradle.startParameter.taskNames.any { task ->
    task.contains("DownloadableCore", ignoreCase = true)
}
if (downloadableCoreTaskRequested && liteRtApiAar == null) {
    throw GradleException(
        "Downloadable-core builds require -PliteRtApiAar=<verified pure API AAR>."
    )
}
if (liteRtApiAar != null) {
    if (!liteRtApiAar.isFile) {
        throw GradleException("LiteRT pure API AAR was not found: $liteRtApiAar")
    }
    val digest = MessageDigest.getInstance("SHA-256")
    val actualSha256 = liteRtApiAar.inputStream().buffered().use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }
    if (actualSha256 != expectedLiteRtApiAarSha256) {
        throw GradleException(
            "LiteRT pure API AAR SHA-256 mismatch: $actualSha256 != $expectedLiteRtApiAarSha256"
        )
    }
}

android {
    namespace = "com.example.musicsourceseparation"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.example.musicsourceseparation"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = true
    }

    flavorDimensions += "runtime"
    productFlavors {
        create("standard") {
            dimension = "runtime"
            buildConfigField("boolean", "DOWNLOADABLE_LITERT_CORE", "false")
        }
        create("downloadableCore") {
            dimension = "runtime"
            buildConfigField("boolean", "DOWNLOADABLE_LITERT_CORE", "true")
        }
    }

    sourceSets {
        getByName("main") {
            assets.directories.add("../models/uvr-mdx")
        }
    }
}

dependencies {
    // 2.1.6 currently publishes litert and litert-api with the same AAR namespace,
    // which AGP 9.2 rejects. 2.1.5 contains the same CompiledModel GPU API in one AAR.
    add("standardImplementation", "com.google.ai.edge.litert:litert:2.1.5")
    if (liteRtApiAar != null) {
        add("downloadableCoreImplementation", files(liteRtApiAar))
    }
    // Historical x86 benchmark artifact. Canonical builds are published by
    // https://github.com/WluhWluh/bss-litert-android.
    add("standardImplementation", files("libs/litert-2.1.5-x86.aar"))
    implementation("com.github.wendykierp:JTransforms:3.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    testImplementation("junit:junit:4.13.2")
}
