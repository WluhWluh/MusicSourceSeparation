package com.example.musicsourceseparation

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import com.example.musicsourceseparation.audio.AudioMetadata
import com.example.musicsourceseparation.audio.AudioMetadataReader
import com.example.musicsourceseparation.audio.AudioPassthroughExporter
import com.example.musicsourceseparation.benchmark.applive.AppLiveRunOrigin
import com.example.musicsourceseparation.benchmark.applive.AppLiveValidationBundle
import com.example.musicsourceseparation.benchmark.applive.AppLiveValidationProfile
import com.example.musicsourceseparation.benchmark.applive.AppLiveValidationService
import com.example.musicsourceseparation.benchmark.applive.AppLiveValidationState
import com.example.musicsourceseparation.model.MdxOnnxSmokeTester
import com.example.musicsourceseparation.model.MdxModelVariant
import com.example.musicsourceseparation.model.MdxOneWindowSeparator
import com.example.musicsourceseparation.model.MdxRangeSeparator
import com.example.musicsourceseparation.model.MdxRuntimeSettings

class MainActivity : Activity() {
    private lateinit var appLiveBundleText: TextView
    private lateinit var appLiveStatusText: TextView
    private lateinit var appLiveOriginSpinner: Spinner
    private lateinit var appLiveQuickButton: Button
    private lateinit var appLiveFullButton: Button
    private lateinit var selectedFileText: TextView
    private lateinit var statusText: TextView
    private lateinit var exportButton: Button
    private lateinit var separateRangeButton: Button
    private lateinit var separateOneWindowButton: Button
    private lateinit var separate37sWindowButton: Button
    private lateinit var onnxSmokeTestButton: Button
    private lateinit var startMinutesInput: EditText
    private lateinit var startSecondsInput: EditText
    private lateinit var startMillisInput: EditText
    private lateinit var endMinutesInput: EditText
    private lateinit var endSecondsInput: EditText
    private lateinit var endMillisInput: EditText
    private lateinit var modelSpinner: Spinner
    private lateinit var cpuThreadsInput: EditText
    private lateinit var useXnnpackInput: CheckBox
    private var selectedAudioUri: Uri? = null
    private var selectedAudioMetadata: AudioMetadata? = null
    private val appLiveHandler = Handler(Looper.getMainLooper())
    private val appLiveRefresh = object : Runnable {
        override fun run() {
            refreshAppLiveControls()
            appLiveHandler.postDelayed(this, APP_LIVE_REFRESH_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val snapshot = AppLiveValidationState.snapshot(this)
        if (snapshot.running && !AppLiveValidationService.isRunning()) {
            AppLiveValidationState.markInterrupted(this)
        }
        setContentView(createContentView())
        refreshAppLiveControls()
    }

    override fun onResume() {
        super.onResume()
        appLiveHandler.removeCallbacks(appLiveRefresh)
        appLiveHandler.post(appLiveRefresh)
    }

    override fun onPause() {
        appLiveHandler.removeCallbacks(appLiveRefresh)
        super.onPause()
    }

    @Deprecated("The platform callback is sufficient for this dependency-light scaffold.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_AUDIO || resultCode != RESULT_OK) return

        val uri = data?.data ?: return
        val persistableGranted = data.flags and Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION != 0
        val readPermissionGranted = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION != 0
        if (persistableGranted && readPermissionGranted) {
            runCatching {
                contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }

        selectedAudioUri = uri
        selectedFileText.text = getString(R.string.reading_audio)
        statusText.text = ""
        exportButton.isEnabled = false
        Thread {
            val result = runCatching { AudioMetadataReader(this).read(uri) }
            runOnUiThread {
                result.onSuccess { metadata ->
                    selectedAudioMetadata = metadata
                    selectedFileText.text = metadata.toDisplayText()
                    setDefaultRange(metadata.durationMs)
                    statusText.text = getString(R.string.ready_for_pipeline)
                    exportButton.isEnabled = true
                    separateRangeButton.isEnabled = true
                    separateOneWindowButton.isEnabled = true
                    separate37sWindowButton.isEnabled = true
                }.onFailure { error ->
                    selectedAudioMetadata = null
                    selectedFileText.text = getString(R.string.no_file_selected)
                    statusText.text = error.message ?: getString(R.string.export_failed)
                }
            }
        }.start()
    }

    private fun createContentView(): ScrollView {
        val density = resources.displayMetrics.density
        val padding = (24 * density).toInt()

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(padding, padding, padding, padding)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            )
        }

        val title = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 24f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
        }
        val appLiveControls = createAppLiveControls()

        selectedFileText = TextView(this).apply {
            text = getString(R.string.no_file_selected)
            textSize = 16f
            gravity = Gravity.CENTER
        }

        statusText = TextView(this).apply {
            text = ""
            textSize = 14f
            gravity = Gravity.CENTER
        }

        val selectButton = Button(this).apply {
            text = getString(R.string.select_audio)
            setOnClickListener { openAudioPicker() }
        }

        exportButton = Button(this).apply {
            text = getString(R.string.export_wav)
            isEnabled = false
            setOnClickListener { exportSelectedAudio() }
        }

        val rangeInputs = createRangeInputs()
        val modelInputs = createModelInputs()
        val runtimeInputs = createRuntimeInputs()

        separateRangeButton = Button(this).apply {
            text = getString(R.string.separate_range)
            isEnabled = false
            setOnClickListener { separateRange() }
        }

        separateOneWindowButton = Button(this).apply {
            text = getString(R.string.separate_one_window)
            isEnabled = false
            setOnClickListener { separateOneWindow(startSeconds = 0.0) }
        }

        separate37sWindowButton = Button(this).apply {
            text = getString(R.string.separate_37s_window)
            isEnabled = false
            setOnClickListener { separateOneWindow(startSeconds = 37.0) }
        }

        onnxSmokeTestButton = Button(this).apply {
            text = getString(R.string.run_onnx_smoke_test)
            setOnClickListener { runOnnxSmokeTest() }
        }

        container.addView(title, spacedLayoutParams(top = 16, density = density))
        container.addView(appLiveControls, spacedLayoutParams(top = 20, density = density))
        container.addView(selectedFileText, spacedLayoutParams(top = 28, density = density))
        container.addView(selectButton, spacedLayoutParams(top = 20, density = density))
        container.addView(rangeInputs, spacedLayoutParams(top = 20, density = density))
        container.addView(modelInputs, spacedLayoutParams(top = 12, density = density))
        container.addView(runtimeInputs, spacedLayoutParams(top = 12, density = density))
        container.addView(separateRangeButton, spacedLayoutParams(top = 12, density = density))
        container.addView(exportButton, spacedLayoutParams(top = 12, density = density))
        container.addView(separateOneWindowButton, spacedLayoutParams(top = 12, density = density))
        container.addView(separate37sWindowButton, spacedLayoutParams(top = 12, density = density))
        container.addView(onnxSmokeTestButton, spacedLayoutParams(top = 12, density = density))
        container.addView(statusText, spacedLayoutParams(top = 20, density = density))

        return ScrollView(this).apply {
            addView(container)
        }
    }

    private fun createAppLiveControls(): LinearLayout {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val title = TextView(this).apply {
            text = getString(R.string.app_live_validation_title)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
        }
        appLiveBundleText = TextView(this).apply {
            textSize = 13f
        }
        val originLabel = TextView(this).apply {
            text = getString(R.string.app_live_run_origin)
            textSize = 14f
        }
        appLiveOriginSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                AppLiveRunOrigin.entries.map { it.displayName },
            )
            val currentOrigin = AppLiveValidationState.snapshot(this@MainActivity).origin
            setSelection(AppLiveRunOrigin.entries.indexOf(currentOrigin))
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                    AppLiveValidationState.setOrigin(
                        this@MainActivity,
                        AppLiveRunOrigin.entries[position],
                    )
                }

                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }
        }
        appLiveQuickButton = Button(this).apply {
            text = getString(R.string.app_live_quick_gate)
            setOnClickListener { startAppLiveValidation(AppLiveValidationProfile.QUICK) }
        }
        appLiveFullButton = Button(this).apply {
            text = getString(R.string.app_live_full_validation)
            setOnClickListener { confirmFullValidation() }
        }
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(appLiveQuickButton, LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ).apply { marginEnd = (8 * resources.displayMetrics.density).toInt() })
            addView(appLiveFullButton, LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f,
            ))
        }
        appLiveStatusText = TextView(this).apply {
            textSize = 13f
        }
        root.addView(title)
        root.addView(appLiveBundleText, spacedLayoutParams(top = 6, density = resources.displayMetrics.density))
        root.addView(originLabel, spacedLayoutParams(top = 10, density = resources.displayMetrics.density))
        root.addView(appLiveOriginSpinner)
        root.addView(buttonRow, spacedLayoutParams(top = 8, density = resources.displayMetrics.density))
        root.addView(appLiveStatusText, spacedLayoutParams(top = 8, density = resources.displayMetrics.density))
        return root
    }

    private fun confirmFullValidation() {
        AlertDialog.Builder(this)
            .setTitle(R.string.app_live_full_validation)
            .setMessage(R.string.app_live_full_validation_confirmation)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.app_live_run) { _, _ ->
                startAppLiveValidation(AppLiveValidationProfile.FULL)
            }
            .show()
    }

    private fun startAppLiveValidation(profile: AppLiveValidationProfile) {
        val bundle = AppLiveValidationBundle.loadCatching(this)
        if (bundle.isFailure) {
            appLiveStatusText.text = bundle.exceptionOrNull()?.message ?: getString(R.string.app_live_bundle_missing)
            return
        }
        val snapshot = AppLiveValidationState.snapshot(this)
        if (snapshot.running || AppLiveValidationService.isRunning()) return
        AppLiveValidationState.update(
            this,
            state = "starting",
            running = true,
            profile = profile,
            runId = "",
            message = "Starting ${profile.displayName}",
        )
        runCatching { AppLiveValidationService.start(this, profile) }
            .onFailure { error ->
                AppLiveValidationState.update(
                    this,
                    state = "error",
                    running = false,
                    profile = profile,
                    runId = "",
                    message = error.message ?: error::class.java.simpleName,
                )
            }
        refreshAppLiveControls()
    }

    private fun refreshAppLiveControls() {
        if (!::appLiveBundleText.isInitialized) return
        val bundleResult = AppLiveValidationBundle.loadCatching(this)
        val snapshot = AppLiveValidationState.snapshot(this)
        val running = snapshot.running || AppLiveValidationService.isRunning()
        bundleResult.onSuccess { bundle ->
            appLiveBundleText.text = getString(
                R.string.app_live_bundle_ready,
                bundle.bundleId.take(12),
                bundle.diagnosticContractVersion,
                bundle.relay.campaign,
            ) + "\n" + getString(
                if (snapshot.origin == AppLiveRunOrigin.BROWSERSTACK) {
                    R.string.app_live_large_artifacts_enabled
                } else {
                    R.string.app_live_large_artifacts_local
                },
            )
        }.onFailure { error ->
            appLiveBundleText.text = error.message ?: getString(R.string.app_live_bundle_missing)
        }
        appLiveQuickButton.isEnabled = bundleResult.isSuccess && !running
        appLiveFullButton.isEnabled = bundleResult.isSuccess && !running
        appLiveOriginSpinner.isEnabled = !running
        appLiveStatusText.text = buildString {
            append(snapshot.state.uppercase())
            append(": ")
            append(snapshot.message)
            if (snapshot.runId.isNotBlank()) {
                append("\n")
                append(snapshot.runId)
            }
        }
    }

    private fun createRangeInputs(): LinearLayout {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        val startLabel = TextView(this).apply {
            text = "Start"
            textSize = 14f
        }
        startMinutesInput = numberInput("min", "0")
        startSecondsInput = numberInput("sec", "0")
        startMillisInput = numberInput("ms", "0")

        val endLabel = TextView(this).apply {
            text = "End"
            textSize = 14f
        }
        endMinutesInput = numberInput("min", "0")
        endSecondsInput = numberInput("sec", "0")
        endMillisInput = numberInput("ms", "0")

        root.addView(startLabel)
        root.addView(horizontalInputs(startMinutesInput, startSecondsInput, startMillisInput))
        root.addView(endLabel, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { topMargin = (8 * resources.displayMetrics.density).toInt() })
        root.addView(horizontalInputs(endMinutesInput, endSecondsInput, endMillisInput))
        return root
    }

    private fun createRuntimeInputs(): LinearLayout {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val label = TextView(this).apply {
            text = "ONNX CPU threads (0 = default)"
            textSize = 14f
        }
        cpuThreadsInput = numberInput("threads", "8")
        useXnnpackInput = CheckBox(this).apply {
            text = "Use XNNPACK"
            textSize = 14f
        }
        root.addView(label)
        root.addView(horizontalInputs(cpuThreadsInput))
        root.addView(useXnnpackInput)
        return root
    }

    private fun createModelInputs(): LinearLayout {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val label = TextView(this).apply {
            text = "Model"
            textSize = 14f
        }
        modelSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                MdxModelVariant.entries.toList(),
            )
        }
        root.addView(label)
        root.addView(modelSpinner)
        return root
    }

    private fun numberInput(hint: String, value: String): EditText {
        return EditText(this).apply {
            this.hint = hint
            setText(value)
            inputType = InputType.TYPE_CLASS_NUMBER
            textSize = 14f
            setSelectAllOnFocus(true)
        }
    }

    private fun horizontalInputs(vararg inputs: EditText): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            for (input in inputs) {
                addView(input, LinearLayout.LayoutParams(
                    0,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    1f,
                ).apply {
                    marginEnd = (8 * resources.displayMetrics.density).toInt()
                })
            }
        }
    }

    private fun openAudioPicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }
        startActivityForResult(intent, REQUEST_AUDIO)
    }

    private fun exportSelectedAudio() {
        val uri = selectedAudioUri ?: return
        val metadata = selectedAudioMetadata ?: return
        setExportEnabled(false)
        statusText.text = getString(R.string.exporting_wav)

        Thread {
            val result = runCatching {
                AudioPassthroughExporter(this).exportToWav(uri, metadata.displayName)
            }
            runOnUiThread {
                result.onSuccess { export ->
                    statusText.text = getString(R.string.export_complete) + ": " + export.outputFile.absolutePath
                }.onFailure { error ->
                    statusText.text = getString(R.string.export_failed) + ": " + (error.message ?: error::class.java.simpleName)
                }
                setExportEnabled(true)
            }
        }.start()
    }

    private fun setExportEnabled(enabled: Boolean) {
        exportButton.isEnabled = enabled && selectedAudioUri != null && selectedAudioMetadata != null
    }

    private fun separateRange() {
        val uri = selectedAudioUri ?: return
        val metadata = selectedAudioMetadata ?: return
        val startMs = readTimeMs(startMinutesInput, startSecondsInput, startMillisInput)
        val endMs = readTimeMs(endMinutesInput, endSecondsInput, endMillisInput).takeIf { it > 0L }
        val modelVariant = selectedModelVariant()
        val runtimeSettings = MdxRuntimeSettings(
            cpuThreads = readCpuThreads(),
            useXnnpack = useXnnpackInput.isChecked,
        )
        setSeparationButtonsEnabled(false)
        statusText.text = getString(R.string.range_separating) +
            "\nModel: ${modelVariant.displayName}\n" +
            runtimeSettings.toDisplayText()

        Thread {
            val result = runCatching {
                MdxRangeSeparator(this).separate(
                    uri = uri,
                    displayName = metadata.displayName,
                    startMs = startMs,
                    endMs = endMs,
                    runtimeSettings = runtimeSettings,
                    modelVariant = modelVariant,
                ) { progress ->
                    runOnUiThread {
                        statusText.text = getString(R.string.range_separating) +
                            ": ${progress.completedWindows}/${progress.totalWindows} (${progress.percent}%)\n" +
                            "Model: ${modelVariant.displayName}\n" +
                            runtimeSettings.toDisplayText()
                    }
                }
            }
            runOnUiThread {
                statusText.text = result.fold(
                    onSuccess = { it.toDisplayText() },
                    onFailure = {
                        getString(R.string.range_failed) + ": " +
                            (it.message ?: it::class.java.simpleName)
                    },
                )
                setSeparationButtonsEnabled(selectedAudioUri != null && selectedAudioMetadata != null)
            }
        }.start()
    }

    private fun separateOneWindow(startSeconds: Double) {
        val uri = selectedAudioUri ?: return
        val metadata = selectedAudioMetadata ?: return
        val modelVariant = selectedModelVariant()
        separateOneWindowButton.isEnabled = false
        separate37sWindowButton.isEnabled = false
        statusText.text = getString(R.string.one_window_separating) + "\nModel: ${modelVariant.displayName}"

        Thread {
            val result = runCatching {
                MdxOneWindowSeparator(this).separate(
                    uri = uri,
                    displayName = metadata.displayName,
                    startSeconds = startSeconds,
                    modelVariant = modelVariant,
                )
            }
            runOnUiThread {
                statusText.text = result.fold(
                    onSuccess = { it.toDisplayText() },
                    onFailure = {
                        getString(R.string.one_window_failed) + ": " +
                            (it.message ?: it::class.java.simpleName)
                    },
                )
                separateOneWindowButton.isEnabled = selectedAudioUri != null && selectedAudioMetadata != null
                separate37sWindowButton.isEnabled = selectedAudioUri != null && selectedAudioMetadata != null
            }
        }.start()
    }

    private fun setSeparationButtonsEnabled(enabled: Boolean) {
        separateRangeButton.isEnabled = enabled
        separateOneWindowButton.isEnabled = enabled
        separate37sWindowButton.isEnabled = enabled
    }

    private fun setDefaultRange(durationMs: Long?) {
        writeTime(startMinutesInput, startSecondsInput, startMillisInput, 0L)
        writeTime(endMinutesInput, endSecondsInput, endMillisInput, durationMs ?: 0L)
    }

    private fun readTimeMs(minutes: EditText, seconds: EditText, millis: EditText): Long {
        val minuteValue = minutes.text.toString().toLongOrNull() ?: 0L
        val secondValue = seconds.text.toString().toLongOrNull() ?: 0L
        val millisValue = millis.text.toString().toLongOrNull() ?: 0L
        return minuteValue * 60_000L + secondValue * 1000L + millisValue
    }

    private fun readCpuThreads(): Int {
        return cpuThreadsInput.text.toString().toIntOrNull()?.coerceIn(0, 16) ?: 0
    }

    private fun selectedModelVariant(): MdxModelVariant {
        return modelSpinner.selectedItem as? MdxModelVariant ?: MdxModelVariant.INST_MAIN
    }

    private fun writeTime(minutes: EditText, seconds: EditText, millis: EditText, totalMs: Long) {
        val safeMs = totalMs.coerceAtLeast(0L)
        minutes.setText((safeMs / 60_000L).toString())
        seconds.setText(((safeMs % 60_000L) / 1000L).toString())
        millis.setText((safeMs % 1000L).toString())
    }

    private fun runOnnxSmokeTest() {
        val modelVariant = selectedModelVariant()
        onnxSmokeTestButton.isEnabled = false
        statusText.text = getString(R.string.onnx_smoke_testing) + "\nModel: ${modelVariant.displayName}"

        Thread {
            val result = runCatching { MdxOnnxSmokeTester(this).run(modelVariant) }
            runOnUiThread {
                statusText.text = result.fold(
                    onSuccess = { it.toDisplayText() },
                    onFailure = {
                        getString(R.string.onnx_smoke_test_failed) + ": " +
                            (it.message ?: it::class.java.simpleName)
                    },
                )
                onnxSmokeTestButton.isEnabled = true
            }
        }.start()
    }

    private fun spacedLayoutParams(top: Int, density: Float): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply {
            topMargin = (top * density).toInt()
        }
    }

    private companion object {
        const val REQUEST_AUDIO = 1001
        const val APP_LIVE_REFRESH_MS = 500L
    }
}
