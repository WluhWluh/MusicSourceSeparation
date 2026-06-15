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
    implementation("com.github.wendykierp:JTransforms:3.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    testImplementation("junit:junit:4.13.2")
}
