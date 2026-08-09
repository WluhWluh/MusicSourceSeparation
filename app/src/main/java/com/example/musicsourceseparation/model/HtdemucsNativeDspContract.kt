package com.example.musicsourceseparation.model

data class HtdemucsNativeDspNormalization(
    val window: String,
    val forwardScale: String,
    val inverseScale: String,
    val outerPadLeft: Int,
    val framePadLeft: Int,
    val framePadRight: Int,
    val centerTrim: Int,
)

data class HtdemucsNativeDspContract(
    val variant: String,
    val executableContractId: String,
    val modelId: String,
    val artifactFileName: String,
    val artifactByteSize: Long,
    val artifactSha256: String,
    val sourceOrder: List<String>,
    val researchOnly: Boolean,
    val licenseStatus: String,
    val sampleRate: Int = 44_100,
    val channelCount: Int = 2,
    val windowSamples: Int = 343_980,
    val spectrumFrames: Int = 336,
    val fftSize: Int = 4_096,
    val hopSize: Int = 1_024,
    val dimF: Int = 2_048,
    val featureCount: Int = 4,
    val normalization: HtdemucsNativeDspNormalization = CANONICAL_NORMALIZATION,
) {
    val sourceCount: Int = sourceOrder.size
    val inputShapes: List<List<Int>> = listOf(
        listOf(1, channelCount, windowSamples),
        listOf(1, featureCount, dimF, spectrumFrames),
    )
    val outputShapes: List<List<Int>> = listOf(
        listOf(1, sourceCount, featureCount, dimF, spectrumFrames),
        listOf(1, sourceCount, channelCount, windowSamples),
    )
    val waveformInputElements: Int = channelCount * windowSamples
    val spectrumInputElements: Int = featureCount * dimF * spectrumFrames
    val frequencyOutputElements: Int = sourceCount * spectrumInputElements
    val waveformOutputElements: Int = sourceCount * channelCount * windowSamples

    init {
        require(CONTRACT_ID.matches(executableContractId))
        require(MODEL_ID.matches(modelId))
        require(SHA256.matches(artifactSha256))
        require(artifactByteSize > 0L)
        require(artifactFileName.endsWith(".tflite"))
        require(sourceOrder == FOUR_STEM_ORDER || sourceOrder == SIX_STEM_ORDER) {
            "Native HTDemucs DSP supports only the frozen four- and six-stem orders."
        }
        require(sampleRate == 44_100 && channelCount == 2)
        require(windowSamples == 343_980 && spectrumFrames == 336)
        require(fftSize == 4_096 && hopSize == 1_024 && dimF == 2_048)
        require(featureCount == channelCount * 2)
        require(spectrumFrames == (windowSamples + hopSize - 1) / hopSize)
        require(normalization == CANONICAL_NORMALIZATION)
        require(!researchOnly || licenseStatus == "research-only-license-review-required")
    }

    companion object {
        val FOUR_STEM_ORDER = listOf("drums", "bass", "other", "vocals")
        val SIX_STEM_ORDER = FOUR_STEM_ORDER + listOf("guitar", "piano")
        val CANONICAL_NORMALIZATION = HtdemucsNativeDspNormalization(
            window = "periodic-hann",
            forwardScale = "divide-by-sqrt-fft-size",
            inverseScale = "multiply-by-sqrt-fft-size-after-normalized-inverse",
            outerPadLeft = 1_536,
            framePadLeft = 2,
            framePadRight = 2,
            centerTrim = 2_048,
        )

        private val CONTRACT_ID = Regex("^[a-z0-9_]+@[1-9][0-9]*$")
        private val MODEL_ID = Regex("^[a-z0-9_]+$")
        private val SHA256 = Regex("^[0-9a-f]{64}$")
    }
}

object HtdemucsNativeDspContracts {
    const val VARIANT_OFFICIAL_6S = "official"
    const val VARIANT_OFFICIAL_4S = "official-4s"
    const val VARIANT_GUITAR_FT_6S = "guitar-ft"

    val OFFICIAL_6S = HtdemucsNativeDspContract(
        variant = VARIANT_OFFICIAL_6S,
        executableContractId = "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0@1",
        modelId = "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0",
        artifactFileName = "htdemucs_6s.core.canonical_7p8s.fp32.tflite",
        artifactByteSize = 117_624_880L,
        artifactSha256 = "8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7",
        sourceOrder = HtdemucsNativeDspContract.SIX_STEM_ORDER,
        researchOnly = false,
        licenseStatus = "mit-product-review-complete",
    )

    val OFFICIAL_4S = HtdemucsNativeDspContract(
        variant = VARIANT_OFFICIAL_4S,
        executableContractId = "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0@1",
        modelId = "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0",
        artifactFileName = "htdemucs_4s.core.canonical_7p8s.fp32.tflite",
        artifactByteSize = 178_042_000L,
        artifactSha256 = "9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81",
        sourceOrder = HtdemucsNativeDspContract.FOUR_STEM_ORDER,
        researchOnly = false,
        licenseStatus = "mit-product-review-complete",
    )

    val GUITAR_FT_6S = HtdemucsNativeDspContract(
        variant = VARIANT_GUITAR_FT_6S,
        executableContractId = "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0@1",
        modelId = "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0",
        artifactFileName = "htdemucs_6s_guitar_ft.core.canonical_7p8s.fp32.tflite",
        artifactByteSize = 117_729_544L,
        artifactSha256 = "ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5",
        sourceOrder = HtdemucsNativeDspContract.SIX_STEM_ORDER,
        researchOnly = true,
        licenseStatus = "research-only-license-review-required",
    )

    val ALL: List<HtdemucsNativeDspContract> = listOf(OFFICIAL_6S, OFFICIAL_4S, GUITAR_FT_6S)
    private val BY_VARIANT = ALL.associateBy(HtdemucsNativeDspContract::variant)

    fun forVariant(variant: String): HtdemucsNativeDspContract =
        requireNotNull(BY_VARIANT[variant]) { "Unsupported HTDemucs variant '$variant'." }
}
