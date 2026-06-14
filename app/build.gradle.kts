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
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
}
