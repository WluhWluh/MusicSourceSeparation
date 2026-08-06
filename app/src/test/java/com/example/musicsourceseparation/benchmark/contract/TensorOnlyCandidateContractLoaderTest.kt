package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.nio.charset.StandardCharsets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class TensorOnlyCandidateContractLoaderTest {
    @Test
    fun loadsCanonicalFourStemConversionSource() {
        val loaded = TensorOnlyCandidateContractLoader.load(
            fourStemSidecar(),
            6_360L,
            FOUR_STEM_SIDECAR_SHA256,
        )
        val contract = loaded.contract

        assertEquals(3, contract.contractSchemaVersion)
        assertEquals("htdemucs_4s_waveform_7p8s_onnx@3", contract.contractId)
        assertEquals("tensor-only-candidate", contract.contractKind)
        assertEquals(TensorOnlyArtifactFormat.ONNX_MODEL, contract.conversionSource.format)
        assertEquals(165_612_636L, contract.conversionSource.byteSize)
        assertEquals(84_141_911L, contract.conversion.exporterInput.byteSize)
        assertEquals(TensorOnlyArtifactFormat.MARKDOWN, contract.conversion.modelCard.format)
        assertEquals("MIT", contract.conversion.modelLicenseSpdx)
        assertEquals("not-established", contract.precisionContract.runtimePrecisionStatus)
        assertEquals(148, contract.precisionContract.initializerStorage.first { it.dtype == "float16" }.tensorCount)
        assertEquals("adefossez/HTDemucs", contract.upstream.hubRepository)
        assertEquals(84_025_440L, contract.upstream.weight.byteSize)
        assertEquals(44_100, contract.workloadContract.sampleRate)
        assertEquals(343_980, contract.workloadContract.sampleCount)
        assertEquals(listOf(1, 2, 343_980), contract.tensorContract.inputs.single().shape)
        assertEquals(687_960L, contract.tensorContract.inputs.single().elementCount)
        assertEquals(2_751_840L, contract.tensorContract.inputs.single().byteSize)
        assertEquals(listOf(1, 4, 2, 343_980), contract.tensorContract.outputs.single().shape)
        assertEquals(2_751_840L, contract.tensorContract.outputs.single().elementCount)
        assertEquals(11_007_360L, contract.tensorContract.outputs.single().byteSize)
        assertEquals(
            listOf(
                BenchmarkStemSemantic.DRUMS,
                BenchmarkStemSemantic.BASS,
                BenchmarkStemSemantic.OTHER,
                BenchmarkStemSemantic.VOCALS,
            ),
            contract.stems.map(BenchmarkStem::semantic),
        )
        assertEquals(FOUR_STEM_SIDECAR_SHA256, loaded.sidecarSha256)
    }

    @Test
    fun loadsCanonicalSixStemConversionSourceAndPreservesOutputOrder() {
        val loaded = TensorOnlyCandidateContractLoader.load(sixStemSidecar())
        val contract = loaded.contract

        assertEquals("htdemucs_6s_waveform_7p8s_onnx@3", contract.contractId)
        assertEquals(136_428_532L, contract.conversionSource.byteSize)
        assertEquals(54_996_327L, contract.conversion.exporterInput.byteSize)
        assertEquals(1066L, contract.conversion.repositoryLicense.byteSize)
        assertEquals(144, contract.precisionContract.initializerStorage.first { it.dtype == "float16" }.tensorCount)
        assertEquals("adefossez/HTDemucs-6s", contract.upstream.hubRepository)
        assertEquals(54_885_744L, contract.upstream.weight.byteSize)
        assertEquals(listOf(1, 6, 2, 343_980), contract.tensorContract.outputs.single().shape)
        assertEquals(4_127_760L, contract.tensorContract.outputs.single().elementCount)
        assertEquals(16_511_040L, contract.tensorContract.outputs.single().byteSize)
        assertEquals(
            listOf(
                BenchmarkStemSemantic.DRUMS,
                BenchmarkStemSemantic.BASS,
                BenchmarkStemSemantic.OTHER,
                BenchmarkStemSemantic.VOCALS,
                BenchmarkStemSemantic.GUITAR,
                BenchmarkStemSemantic.PIANO,
            ),
            contract.stems.map(BenchmarkStem::semantic),
        )
        assertEquals(SIX_STEM_SIDECAR_SHA256, loaded.sidecarSha256)
    }

    @Test
    fun acceptsOneWholeTensorOutputPerStem() {
        val contract = TensorOnlyCandidateContractLoader.parse(multiOutputJson())

        assertEquals(4, contract.tensorContract.outputs.size)
        assertEquals(listOf(0, 1, 2, 3), contract.tensorContract.outputs.map(TensorOnlyTensor::index))
        assertEquals(
            List(4) { TensorOnlyStemPackingKind.WHOLE_TENSOR },
            contract.stemContract.bindings.map { it.packing.kind },
        )
        assertEquals(4, contract.stems.size)
    }

    @Test
    fun acceptsNeuralCoreBranchesThatShareOneLogicalStemOrder() {
        val contract = TensorOnlyCandidateContractLoader.parse(neuralCoreJson())

        assertEquals(TensorOnlyWorkloadKind.NEURAL_CORE_SEGMENT, contract.workloadContract.kind)
        assertEquals(2, contract.tensorContract.inputs.size)
        assertEquals(2, contract.tensorContract.outputs.size)
        assertEquals(4, contract.stems.size)
        assertEquals(
            contract.stemContract.bindings[0].stems,
            contract.stemContract.bindings[1].stems,
        )
    }

    @Test
    fun rejectsOutOfOrderOutputBindings() {
        val firstMoved = replaceRequired(
            multiOutputJson(),
            "\"outputIndex\": 0",
            "\"outputIndex\": 99",
        )
        val secondMoved = replaceRequired(firstMoved, "\"outputIndex\": 1", "\"outputIndex\": 0")
        val tampered = replaceRequired(secondMoved, "\"outputIndex\": 99", "\"outputIndex\": 1")

        assertRejected(tampered, "output-index order")
    }

    @Test
    fun rejectsRuntimeContractKind() {
        val tampered = replaceRequired(
            fourStemJson(),
            "\"contractKind\": \"tensor-only-candidate\"",
            "\"contractKind\": \"tensor-only-runtime\"",
        )

        assertRejected(tampered, "contractKind must equal 'tensor-only-candidate'")
    }

    @Test
    fun rejectsPackedStemAxisLengthMismatch() {
        val tampered = replaceRequired(
            fourStemJson(),
            "          4,\n          2,\n          343980",
            "          5,\n          2,\n          343980",
        )

        assertRejected(tampered, "packed axis must equal stem count times elementsPerStem")
    }

    @Test
    fun rejectsDuplicateStemSemantic() {
        val tampered = replaceRequired(
            fourStemJson(),
            "\"semantic\": \"bass\"",
            "\"semantic\": \"drums\"",
        )

        assertRejected(tampered, "stem semantics must be unique")
    }

    @Test
    fun rejectsNonNormalizedArtifactPath() {
        val tampered = replaceRequired(
            fourStemJson(),
            "\"localPath\": \"onnx/htdemucs_fp16weights.onnx\"",
            "\"localPath\": \"../htdemucs_fp16weights.onnx\"",
        )

        assertRejected(tampered, "localPath is not normalized")
    }

    @Test
    fun rejectsOpaqueHttpsArtifactUri() {
        val firstUrl = Regex("\\\"url\\\": \\\"[^\\\"]+\\\"").find(fourStemJson())
            ?: error("Fixture did not contain an artifact URL.")
        val tampered = replaceRequired(
            fourStemJson(),
            firstUrl.value,
            "\"url\": \"https:artifact.onnx\"",
        )

        assertRejected(tampered, "hierarchical HTTPS URI with a host")
    }

    @Test
    fun pinnedSidecarRejectsTampering() {
        val sidecar = File.createTempFile("tampered-htdemucs-", ".json")
        try {
            sidecar.writeText(
                replaceRequired(fourStemJson(), "\"displayName\": \"HTDemucs", "\"displayName\": \"Tampered HTDemucs"),
                StandardCharsets.UTF_8,
            )

            val error = assertThrows(BenchmarkModelContractException::class.java) {
                TensorOnlyCandidateContractLoader.load(sidecar, FOUR_STEM_SIDECAR_SHA256)
            }
            assertTrue(error.message.orEmpty().contains("SHA-256 mismatch"))
        } finally {
            sidecar.delete()
        }
    }

    private fun multiOutputJson(): String {
        val outputs = """    "outputs": [
      {
        "axes": ["batch", "channel", "sample"],
        "dtype": "float32",
        "index": 0,
        "name": "drums",
        "shape": [1, 2, 343980]
      },
      {
        "axes": ["batch", "channel", "sample"],
        "dtype": "float32",
        "index": 1,
        "name": "bass",
        "shape": [1, 2, 343980]
      },
      {
        "axes": ["batch", "channel", "sample"],
        "dtype": "float32",
        "index": 2,
        "name": "other",
        "shape": [1, 2, 343980]
      },
      {
        "axes": ["batch", "channel", "sample"],
        "dtype": "float32",
        "index": 3,
        "name": "vocals",
        "shape": [1, 2, 343980]
      }
    ]"""
        val bindings = """    "bindings": [
      {
        "outputIndex": 0,
        "packing": {"kind": "whole-tensor"},
        "stems": [{"displayLabel": "Drums", "semantic": "drums"}]
      },
      {
        "outputIndex": 1,
        "packing": {"kind": "whole-tensor"},
        "stems": [{"displayLabel": "Bass", "semantic": "bass"}]
      },
      {
        "outputIndex": 2,
        "packing": {"kind": "whole-tensor"},
        "stems": [{"displayLabel": "Other", "semantic": "other"}]
      },
      {
        "outputIndex": 3,
        "packing": {"kind": "whole-tensor"},
        "stems": [{"displayLabel": "Vocals", "semantic": "vocals"}]
      }
    ],"""
        return replaceSectionRequired(
            replaceSectionRequired(
                fourStemJson(),
                "    \"outputs\": [",
                "\n  },\n  \"upstream\":",
                outputs,
            ),
            "    \"bindings\": [",
            "\n    \"derivation\":",
            bindings,
        )
    }

    private fun neuralCoreJson(): String {
        val inputs = """    "inputs": [
      {
        "axes": ["batch", "channel", "sample"],
        "dtype": "float32",
        "index": 0,
        "name": "waveform",
        "shape": [1, 2, 343980]
      },
      {
        "axes": ["batch", "feature", "frequency", "frame"],
        "dtype": "float32",
        "index": 1,
        "name": "spectrogram",
        "shape": [1, 4, 2048, 336]
      }
    ]"""
        val outputs = """    "outputs": [
      {
        "axes": ["batch", "stem", "channel", "sample"],
        "dtype": "float32",
        "index": 0,
        "name": "time_stems",
        "shape": [1, 4, 2, 343980]
      },
      {
        "axes": ["batch", "stem", "feature", "frequency", "frame"],
        "dtype": "float32",
        "index": 1,
        "name": "spectrogram_stems",
        "shape": [1, 4, 4, 2048, 336]
      }
    ]"""
        val stems = """[
          {"displayLabel": "Drums", "semantic": "drums"},
          {"displayLabel": "Bass", "semantic": "bass"},
          {"displayLabel": "Other", "semantic": "other"},
          {"displayLabel": "Vocals", "semantic": "vocals"}
        ]"""
        val bindings = """    "bindings": [
      {
        "outputIndex": 0,
        "packing": {"axis": 1, "elementsPerStem": 1, "kind": "axis-contiguous"},
        "stems": $stems
      },
      {
        "outputIndex": 1,
        "packing": {"axis": 1, "elementsPerStem": 1, "kind": "axis-contiguous"},
        "stems": $stems
      }
    ],"""
        val withInputs = replaceSectionRequired(
            fourStemJson(),
            "    \"inputs\": [",
            ",\n    \"outputs\": [",
            inputs,
        )
        val withOutputs = replaceSectionRequired(
            withInputs,
            "    \"outputs\": [",
            "\n  },\n  \"upstream\":",
            outputs,
        )
        val withBindings = replaceSectionRequired(
            withOutputs,
            "    \"bindings\": [",
            "\n    \"derivation\":",
            bindings,
        )
        return replaceRequired(
            withBindings,
            "\"kind\": \"waveform-segment\"",
            "\"kind\": \"neural-core-segment\"",
        )
    }

    private fun replaceSectionRequired(
        source: String,
        startMarker: String,
        endMarker: String,
        replacement: String,
    ): String {
        val start = source.indexOf(startMarker)
        check(start >= 0) { "Fixture did not contain start marker '$startMarker'." }
        val end = source.indexOf(endMarker, start)
        check(end > start) { "Fixture did not contain end marker '$endMarker'." }
        return source.replaceRange(start, end, replacement)
    }

    private fun assertRejected(json: String, messageFragment: String) {
        val error = assertThrows(BenchmarkModelContractException::class.java) {
            TensorOnlyCandidateContractLoader.parse(json)
        }
        assertTrue(
            "Expected '${error.message}' to contain '$messageFragment'.",
            error.message.orEmpty().contains(messageFragment),
        )
    }

    private fun fourStemJson(): String = fourStemSidecar().readText(StandardCharsets.UTF_8)

    private fun fourStemSidecar(): File = sidecar("htdemucs_4s_waveform_7p8s_onnx.json")

    private fun sixStemSidecar(): File = sidecar("htdemucs_6s_waveform_7p8s_onnx.json")

    private fun sidecar(name: String): File {
        val relative = "src/main/assets/benchmark-contracts/$name"
        return sequenceOf(File(relative), File("app/$relative"))
            .firstOrNull(File::isFile)
            ?: error("Could not locate tensor-only benchmark contract asset $name.")
    }

    private fun replaceRequired(source: String, oldValue: String, newValue: String): String {
        check(source.contains(oldValue)) { "Fixture did not contain expected mutation target." }
        return source.replaceFirst(oldValue, newValue)
    }

    companion object {
        private const val FOUR_STEM_SIDECAR_SHA256 =
            "f53a3cdad1d162541b8b353257d716e25b4178a433d8629a4804b3178484277d"
        private const val SIX_STEM_SIDECAR_SHA256 =
            "cc8dafb665602b6b8410ca87c6450e29ff4986990f22613e6451cf8da9f4c513"
    }
}
