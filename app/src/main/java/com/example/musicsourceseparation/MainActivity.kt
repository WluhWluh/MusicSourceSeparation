package com.example.musicsourceseparation

import android.app.Activity
import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.example.musicsourceseparation.audio.AudioMetadata
import com.example.musicsourceseparation.audio.AudioMetadataReader
import com.example.musicsourceseparation.audio.AudioPassthroughExporter
import com.example.musicsourceseparation.model.MdxOnnxSmokeTester
import com.example.musicsourceseparation.model.MdxOneWindowSeparator

class MainActivity : Activity() {
    private lateinit var selectedFileText: TextView
    private lateinit var statusText: TextView
    private lateinit var exportButton: Button
    private lateinit var separateOneWindowButton: Button
    private lateinit var separate37sWindowButton: Button
    private lateinit var onnxSmokeTestButton: Button
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
                    statusText.text = getString(R.string.ready_for_pipeline)
                    exportButton.isEnabled = true
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
        container.addView(exportButton, spacedLayoutParams(top = 12, density = density))
        container.addView(separateOneWindowButton, spacedLayoutParams(top = 12, density = density))
        container.addView(separate37sWindowButton, spacedLayoutParams(top = 12, density = density))
        container.addView(onnxSmokeTestButton, spacedLayoutParams(top = 12, density = density))
        container.addView(statusText, spacedLayoutParams(top = 20, density = density))

        return ScrollView(this).apply {
            addView(container)
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
