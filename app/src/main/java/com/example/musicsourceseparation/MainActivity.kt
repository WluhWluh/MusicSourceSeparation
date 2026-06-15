package com.example.musicsourceseparation

import android.app.Activity
import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.example.musicsourceseparation.audio.AudioMetadata
import com.example.musicsourceseparation.audio.AudioMetadataReader
import com.example.musicsourceseparation.audio.AudioPassthroughExporter
import com.example.musicsourceseparation.model.MdxOnnxSmokeTester
import com.example.musicsourceseparation.model.MdxOneWindowSeparator
import com.example.musicsourceseparation.model.MdxRangeSeparator
import com.example.musicsourceseparation.model.MdxRuntimeSettings

class MainActivity : Activity() {
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
    private lateinit var cpuThreadsInput: EditText
    private lateinit var useXnnpackInput: CheckBox
    private var selectedAudioUri: Uri? = null
    private var selectedAudioMetadata: AudioMetadata? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())
    }

    @Deprecated("The platform callback is sufficient for this dependency-light scaffold.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_AUDIO || resultCode != RESULT_OK) return

        val uri = data?.data ?: return
        val persistableGranted = data.flags and Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION != 0
        val readFlags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
        if (persistableGranted && readFlags != 0) {
            runCatching { contentResolver.takePersistableUriPermission(uri, readFlags) }
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
        container.addView(selectedFileText, spacedLayoutParams(top = 28, density = density))
        container.addView(selectButton, spacedLayoutParams(top = 20, density = density))
        container.addView(rangeInputs, spacedLayoutParams(top = 20, density = density))
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
        val runtimeSettings = MdxRuntimeSettings(
            cpuThreads = readCpuThreads(),
            useXnnpack = useXnnpackInput.isChecked,
        )
        setSeparationButtonsEnabled(false)
        statusText.text = getString(R.string.range_separating) + "\n" + runtimeSettings.toDisplayText()

        Thread {
            val result = runCatching {
                MdxRangeSeparator(this).separate(
                    uri = uri,
                    displayName = metadata.displayName,
                    startMs = startMs,
                    endMs = endMs,
                    runtimeSettings = runtimeSettings,
                ) { progress ->
                    runOnUiThread {
                        statusText.text = getString(R.string.range_separating) +
                            ": ${progress.completedWindows}/${progress.totalWindows} (${progress.percent}%)\n" +
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
        separateOneWindowButton.isEnabled = false
        separate37sWindowButton.isEnabled = false
        statusText.text = getString(R.string.one_window_separating)

        Thread {
            val result = runCatching {
                MdxOneWindowSeparator(this).separate(uri, metadata.displayName, startSeconds)
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

    private fun writeTime(minutes: EditText, seconds: EditText, millis: EditText, totalMs: Long) {
        val safeMs = totalMs.coerceAtLeast(0L)
        minutes.setText((safeMs / 60_000L).toString())
        seconds.setText(((safeMs % 60_000L) / 1000L).toString())
        millis.setText((safeMs % 1000L).toString())
    }

    private fun runOnnxSmokeTest() {
        onnxSmokeTestButton.isEnabled = false
        statusText.text = getString(R.string.onnx_smoke_testing)

        Thread {
            val result = runCatching { MdxOnnxSmokeTester(this).run() }
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
    }
}
