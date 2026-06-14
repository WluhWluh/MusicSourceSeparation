package com.example.musicsourceseparation

import android.app.Activity
import android.content.Intent
import android.database.Cursor
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {
    private lateinit var selectedFileText: TextView
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())
    }

    @Deprecated("The platform callback is sufficient for this dependency-light scaffold.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_AUDIO || resultCode != RESULT_OK) return

        val uri = data?.data ?: return
        val flags = data.flags and
            (Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        if (flags and Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION != 0) {
            runCatching { contentResolver.takePersistableUriPermission(uri, flags) }
        }

        selectedFileText.text = getString(R.string.selected_file) + ": " + displayNameFor(uri)
        statusText.text = getString(R.string.ready_for_pipeline)
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

        container.addView(title, spacedLayoutParams(top = 16, density = density))
        container.addView(selectedFileText, spacedLayoutParams(top = 28, density = density))
        container.addView(selectButton, spacedLayoutParams(top = 20, density = density))
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

    private fun displayNameFor(uri: Uri): String {
        val cursor: Cursor? = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            val nameIndex = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex >= 0 && it.moveToFirst()) {
                return it.getString(nameIndex)
            }
        }
        return uri.lastPathSegment ?: uri.toString()
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
