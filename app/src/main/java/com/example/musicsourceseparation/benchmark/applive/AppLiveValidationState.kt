package com.example.musicsourceseparation.benchmark.applive

import android.content.Context

internal enum class AppLiveRunOrigin(val id: String, val displayName: String) {
    BROWSERSTACK("browserstack", "BrowserStack App Live"),
    LOCAL("local", "Local control");

    companion object {
        fun from(value: String?): AppLiveRunOrigin =
            entries.firstOrNull { it.id == value } ?: BROWSERSTACK
    }
}

internal enum class AppLiveValidationProfile(val id: String, val displayName: String) {
    QUICK("quick", "Quick gate"),
    FULL("full", "Full validation"),
    DSP_MATRIX("dsp-matrix", "DSP matrix"),
}

internal data class AppLiveValidationSnapshot(
    val state: String,
    val running: Boolean,
    val origin: AppLiveRunOrigin,
    val profile: String,
    val runId: String,
    val message: String,
    val updatedAtMs: Long,
)

internal object AppLiveValidationState {
    private const val PREFS = "app_live_validation"
    private const val KEY_STATE = "state"
    private const val KEY_RUNNING = "running"
    private const val KEY_ORIGIN = "origin"
    private const val KEY_PROFILE = "profile"
    private const val KEY_RUN_ID = "run_id"
    private const val KEY_MESSAGE = "message"
    private const val KEY_UPDATED_AT = "updated_at"

    fun snapshot(context: Context): AppLiveValidationSnapshot {
        val preferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return AppLiveValidationSnapshot(
            state = preferences.getString(KEY_STATE, "idle") ?: "idle",
            running = preferences.getBoolean(KEY_RUNNING, false),
            origin = AppLiveRunOrigin.from(preferences.getString(KEY_ORIGIN, null)),
            profile = preferences.getString(KEY_PROFILE, "") ?: "",
            runId = preferences.getString(KEY_RUN_ID, "") ?: "",
            message = preferences.getString(KEY_MESSAGE, "Ready") ?: "Ready",
            updatedAtMs = preferences.getLong(KEY_UPDATED_AT, 0L),
        )
    }

    fun setOrigin(context: Context, origin: AppLiveRunOrigin) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_ORIGIN, origin.id)
            .apply()
    }

    fun update(
        context: Context,
        state: String,
        running: Boolean,
        profile: AppLiveValidationProfile,
        runId: String,
        message: String,
    ) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_STATE, state)
            .putBoolean(KEY_RUNNING, running)
            .putString(KEY_PROFILE, profile.id)
            .putString(KEY_RUN_ID, runId)
            .putString(KEY_MESSAGE, message)
            .putLong(KEY_UPDATED_AT, System.currentTimeMillis())
            .apply()
    }

    fun markInterrupted(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_STATE, "error")
            .putBoolean(KEY_RUNNING, false)
            .putString(KEY_MESSAGE, "The previous validation process ended before completion")
            .putLong(KEY_UPDATED_AT, System.currentTimeMillis())
            .apply()
    }
}
