plugins {
    id("com.android.application")
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

    sourceSets {
        getByName("main") {
            assets.directories.add("../models/uvr-mdx")
        }
    }
}

dependencies {
    // 2.1.6 currently publishes litert and litert-api with the same AAR namespace,
    // which AGP 9.2 rejects. 2.1.5 contains the same CompiledModel GPU API in one AAR.
    implementation("com.google.ai.edge.litert:litert:2.1.5")
    // Historical x86 benchmark artifact. Canonical builds are published by
    // https://github.com/WluhWluh/bss-litert-android.
    implementation(files("libs/litert-2.1.5-x86.aar"))
    implementation("com.github.wendykierp:JTransforms:3.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    testImplementation("junit:junit:4.13.2")
}
