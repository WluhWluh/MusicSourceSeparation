package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.math.BigDecimal
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class BenchmarkModelContract(
    val contractSchemaVersion: Int,
    val contractId: String,
    val modelId: String,
    val displayName: String,
    val artifact: BenchmarkArtifactContract,
    val source: BenchmarkSourceContract,
    val conversion: BenchmarkConversionContract,
    val tensorContract: BenchmarkTensorContract,
    val dsp: BenchmarkDspContract,
    val stemContract: BenchmarkStemContract,
    val pipelineCompatibility: BenchmarkPipelineCompatibility,
) {
    val expectedTensorShape: List<Int> = listOf(
        tensorContract.batchSize,
        dsp.dimF,
        dsp.modelTimeFrames,
        tensorContract.complexChannelCount,
    )
    val chunkSizeSamples: Int = Math.multiplyExact(dsp.hopLength, dsp.modelTimeFrames - 1)
    val trimSamples: Int = dsp.nFft / 2
    val generationSizeSamples: Int = Math.subtractExact(
        chunkSizeSamples,
        Math.multiplyExact(2, trimSamples),
    )
}

data class BenchmarkArtifactContract(
    val fileName: String,
    val byteSize: Long,
    val sha256: String,
    val format: String,
)

data class BenchmarkSourceContract(
    val canonicalSourceId: String,
    val fileName: String,
    val url: String,
    val byteSize: Long,
    val sha256: String,
    val attribution: List<String>,
)

data class BenchmarkConversionContract(
    val repository: String,
    val revision: String,
    val pipelineVersion: Int,
    val toolVersions: Map<String, String>,
)

data class BenchmarkTensorContract(
    val input: BenchmarkTensor,
    val output: BenchmarkTensor,
    val batchSize: Int,
    val complexChannelCount: Int,
)

data class BenchmarkTensor(
    val name: String,
    val dtype: String,
    val layout: String,
    val shape: List<Int>,
)

data class BenchmarkDspContract(
    val sampleRate: Int,
    val channelCount: Int,
    val nFft: Int,
    val hopLength: Int,
    val dimF: Int,
    val dimTPower: Int,
    val modelTimeFrames: Int,
    val window: String,
    val modelOutputScale: Double,
)

enum class BenchmarkStemSemantic(val wireValue: String) {
    VOCALS("vocals"),
    INSTRUMENTAL("instrumental"),
    BASS("bass"),
    DRUMS("drums"),
    OTHER("other"),
    REVERB("reverb"),
    NO_CROWD("no_crowd"),
    TARGET_STEM("target_stem"),
    REMAINING_AUDIO("remaining_audio");

    companion object {
        internal fun fromWireValue(value: String, path: String): BenchmarkStemSemantic =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported stem semantic '$value'.")
    }
}

data class BenchmarkStem(
    val semantic: BenchmarkStemSemantic,
    val displayLabel: String,
)

data class BenchmarkStemContract(
    val modelOutput: BenchmarkStem,
    val residual: BenchmarkStem,
    val residualRule: String,
)

data class BenchmarkPipelineCompatibility(
    val pipelineId: String,
    val minimumVersion: Int,
    val maximumVersion: Int,
)

data class LoadedBenchmarkModelContract(
    val contract: BenchmarkModelContract,
    val sidecarFile: File,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
)

class BenchmarkModelContractException(
    message: String,
    cause: Throwable? = null,
) : IllegalArgumentException(message, cause)

object BenchmarkModelContractLoader {
    fun load(sidecarFile: File): LoadedBenchmarkModelContract = loadInternal(sidecarFile, null)

    fun load(sidecarFile: File, expectedSidecarSha256: String): LoadedBenchmarkModelContract {
        expect(
            SHA256.matches(expectedSidecarSha256),
            "Expected sidecar SHA-256 must be 64 lowercase hexadecimal characters.",
        )
        return loadInternal(sidecarFile, expectedSidecarSha256)
    }

    private fun loadInternal(
        sidecarFile: File,
        expectedSidecarSha256: String?,
    ): LoadedBenchmarkModelContract {
        if (!sidecarFile.isFile) {
            invalid("Model contract sidecar is not a readable file: ${sidecarFile.absolutePath}")
        }
        val bytes = try {
            sidecarFile.readBytes()
        } catch (error: Exception) {
            throw BenchmarkModelContractException(
                "Could not read model contract sidecar: ${sidecarFile.absolutePath}",
                error,
            )
        }
        val json = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (error: Exception) {
            throw BenchmarkModelContractException(
                "Model contract sidecar is not valid UTF-8: ${sidecarFile.absolutePath}",
                error,
            )
        }
        val actualSidecarSha256 = sha256(bytes)
        if (expectedSidecarSha256 != null) {
            expect(
                MessageDigest.isEqual(
                    expectedSidecarSha256.toByteArray(StandardCharsets.US_ASCII),
                    actualSidecarSha256.toByteArray(StandardCharsets.US_ASCII),
                ),
                "Model contract sidecar SHA-256 mismatch for ${sidecarFile.absolutePath}.",
            )
        }
        return LoadedBenchmarkModelContract(
            contract = parse(json),
            sidecarFile = sidecarFile.absoluteFile,
            sidecarByteSize = bytes.size.toLong(),
            sidecarSha256 = actualSidecarSha256,
        )
    }

    fun parse(json: String): BenchmarkModelContract {
        val root = ObjectReader(StrictJsonParser(json).parse().asObject("$"), "$")
        root.requireExactKeys(
            "contractSchemaVersion",
            "contractId",
            "modelId",
            "displayName",
            "artifact",
            "source",
            "conversion",
            "tensorContract",
            "dsp",
            "stemContract",
            "pipelineCompatibility",
        )

        val schemaVersion = root.int("contractSchemaVersion")
        expect(schemaVersion == SCHEMA_VERSION, "$.contractSchemaVersion must equal $SCHEMA_VERSION.")
        val contractId = root.string("contractId")
        val modelId = root.string("modelId")
        expect(MODEL_ID.matches(modelId), "$.modelId must match ${MODEL_ID.pattern}.")
        expect(CONTRACT_ID.matches(contractId), "$.contractId must match ${CONTRACT_ID.pattern}.")
        expect(contractId == "$modelId@$SCHEMA_VERSION", "$.contractId must equal '$modelId@$SCHEMA_VERSION'.")

        val contract = BenchmarkModelContract(
            contractSchemaVersion = schemaVersion,
            contractId = contractId,
            modelId = modelId,
            displayName = root.nonEmptyString("displayName"),
            artifact = parseArtifact(root.obj("artifact")),
            source = parseSource(root.obj("source")),
            conversion = parseConversion(root.obj("conversion")),
            tensorContract = parseTensorContract(root.obj("tensorContract")),
            dsp = parseDsp(root.obj("dsp")),
            stemContract = parseStemContract(root.obj("stemContract")),
            pipelineCompatibility = parsePipeline(root.obj("pipelineCompatibility")),
        )
        validateCrossFieldConsistency(contract)
        return contract
    }

    private fun parseArtifact(value: JsonObject): BenchmarkArtifactContract {
        val reader = ObjectReader(value, "$.artifact")
        reader.requireExactKeys("fileName", "byteSize", "sha256", "format")
        val fileName = reader.nonEmptyString("fileName")
        expect(fileName.endsWith(".tflite"), "$.artifact.fileName must end in .tflite.")
        val sha256 = reader.string("sha256")
        expect(SHA256.matches(sha256), "$.artifact.sha256 must be 64 lowercase hexadecimal characters.")
        val format = reader.string("format")
        expect(format == TFLITE_FORMAT, "$.artifact.format must equal '$TFLITE_FORMAT'.")
        return BenchmarkArtifactContract(
            fileName = fileName,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = sha256,
            format = format,
        )
    }

    private fun parseSource(value: JsonObject): BenchmarkSourceContract {
        val reader = ObjectReader(value, "$.source")
        reader.requireExactKeys(
            "canonicalSourceId",
            "fileName",
            "url",
            "byteSize",
            "sha256",
            "attribution",
        )
        val fileName = reader.nonEmptyString("fileName")
        expect(fileName.endsWith(".onnx"), "$.source.fileName must end in .onnx.")
        val url = reader.string("url")
        requireAbsoluteUri(url, "$.source.url")
        val sha256 = reader.string("sha256")
        expect(SHA256.matches(sha256), "$.source.sha256 must be 64 lowercase hexadecimal characters.")
        val attribution = reader.array("attribution").values.mapIndexed { index, item ->
            item.asString("$.source.attribution[$index]").also {
                expect(it.isNotEmpty(), "$.source.attribution[$index] must not be empty.")
            }
        }
        expect(attribution.isNotEmpty(), "$.source.attribution must contain at least one entry.")
        expect(attribution.toSet().size == attribution.size, "$.source.attribution entries must be unique.")
        return BenchmarkSourceContract(
            canonicalSourceId = reader.nonEmptyString("canonicalSourceId"),
            fileName = fileName,
            url = url,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = sha256,
            attribution = attribution,
        )
    }

    private fun parseConversion(value: JsonObject): BenchmarkConversionContract {
        val reader = ObjectReader(value, "$.conversion")
        reader.requireExactKeys("repository", "revision", "pipelineVersion", "toolVersions")
        val repository = reader.string("repository")
        requireAbsoluteUri(repository, "$.conversion.repository")
        val revision = reader.string("revision")
        expect(GIT_REVISION.matches(revision), "$.conversion.revision must be a 40-character lowercase Git revision.")
        val tools = reader.obj("toolVersions")
        expect(tools.values.isNotEmpty(), "$.conversion.toolVersions must not be empty.")
        val toolVersions = tools.values.mapValues { (name, version) ->
            version.asString("$.conversion.toolVersions.$name").also {
                expect(it.isNotEmpty(), "$.conversion.toolVersions.$name must not be empty.")
            }
        }
        return BenchmarkConversionContract(
            repository = repository,
            revision = revision,
            pipelineVersion = reader.positiveInt("pipelineVersion"),
            toolVersions = toolVersions,
        )
    }

    private fun parseTensorContract(value: JsonObject): BenchmarkTensorContract {
        val reader = ObjectReader(value, "$.tensorContract")
        reader.requireExactKeys("input", "output", "batchSize", "complexChannelCount")
        val batchSize = reader.int("batchSize")
        expect(batchSize == 1, "$.tensorContract.batchSize must equal 1.")
        val complexChannelCount = reader.int("complexChannelCount")
        expect(complexChannelCount == 4, "$.tensorContract.complexChannelCount must equal 4.")
        return BenchmarkTensorContract(
            input = parseTensor(reader.obj("input"), "$.tensorContract.input"),
            output = parseTensor(reader.obj("output"), "$.tensorContract.output"),
            batchSize = batchSize,
            complexChannelCount = complexChannelCount,
        )
    }

    private fun parseTensor(value: JsonObject, path: String): BenchmarkTensor {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("name", "dtype", "layout", "shape")
        val dtype = reader.string("dtype")
        expect(dtype == FLOAT32, "$path.dtype must equal '$FLOAT32'.")
        val layout = reader.string("layout")
        expect(layout == NHWC, "$path.layout must equal '$NHWC'.")
        val shape = reader.array("shape").values.mapIndexed { index, dimension ->
            dimension.asPositiveInt("$path.shape[$index]")
        }
        expect(shape.size == 4, "$path.shape must contain exactly four dimensions.")
        return BenchmarkTensor(
            name = reader.nonEmptyString("name"),
            dtype = dtype,
            layout = layout,
            shape = shape,
        )
    }

    private fun parseDsp(value: JsonObject): BenchmarkDspContract {
        val reader = ObjectReader(value, "$.dsp")
        reader.requireExactKeys(
            "sampleRate",
            "channelCount",
            "nFft",
            "hopLength",
            "dimF",
            "dimTPower",
            "modelTimeFrames",
            "window",
            "modelOutputScale",
        )
        val sampleRate = reader.int("sampleRate")
        expect(sampleRate == 44_100, "$.dsp.sampleRate must equal 44100.")
        val channelCount = reader.int("channelCount")
        expect(channelCount == 2, "$.dsp.channelCount must equal 2.")
        val nFft = reader.positiveInt("nFft")
        expect(nFft % 2 == 0, "$.dsp.nFft must be even.")
        val window = reader.string("window")
        expect(window == PERIODIC_HANN, "$.dsp.window must equal '$PERIODIC_HANN'.")
        val outputScale = reader.positiveDouble("modelOutputScale")
        return BenchmarkDspContract(
            sampleRate = sampleRate,
            channelCount = channelCount,
            nFft = nFft,
            hopLength = reader.positiveInt("hopLength"),
            dimF = reader.positiveInt("dimF"),
            dimTPower = reader.positiveInt("dimTPower"),
            modelTimeFrames = reader.positiveInt("modelTimeFrames"),
            window = window,
            modelOutputScale = outputScale,
        )
    }

    private fun parseStemContract(value: JsonObject): BenchmarkStemContract {
        val reader = ObjectReader(value, "$.stemContract")
        reader.requireExactKeys("modelOutput", "residual", "residualRule")
        val residualRule = reader.string("residualRule")
        expect(
            residualRule == RESIDUAL_RULE,
            "$.stemContract.residualRule must equal '$RESIDUAL_RULE'.",
        )
        return BenchmarkStemContract(
            modelOutput = parseStem(reader.obj("modelOutput"), "$.stemContract.modelOutput"),
            residual = parseStem(reader.obj("residual"), "$.stemContract.residual"),
            residualRule = residualRule,
        )
    }

    private fun parseStem(value: JsonObject, path: String): BenchmarkStem {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("semantic", "displayLabel")
        return BenchmarkStem(
            semantic = BenchmarkStemSemantic.fromWireValue(reader.string("semantic"), "$path.semantic"),
            displayLabel = reader.nonEmptyString("displayLabel"),
        )
    }

    private fun parsePipeline(value: JsonObject): BenchmarkPipelineCompatibility {
        val reader = ObjectReader(value, "$.pipelineCompatibility")
        reader.requireExactKeys("pipelineId", "minimumVersion", "maximumVersion")
        val pipelineId = reader.string("pipelineId")
        expect(pipelineId == PIPELINE_ID, "$.pipelineCompatibility.pipelineId must equal '$PIPELINE_ID'.")
        val minimum = reader.positiveInt("minimumVersion")
        val maximum = reader.positiveInt("maximumVersion")
        expect(maximum >= minimum, "$.pipelineCompatibility.maximumVersion must not be below minimumVersion.")
        return BenchmarkPipelineCompatibility(pipelineId, minimum, maximum)
    }

    private fun validateCrossFieldConsistency(contract: BenchmarkModelContract) {
        val dsp = contract.dsp
        expect(dsp.dimTPower < 31, "$.dsp.dimTPower is too large for an Android tensor dimension.")
        expect(
            dsp.modelTimeFrames == (1 shl dsp.dimTPower),
            "$.dsp.modelTimeFrames must equal 2^dimTPower.",
        )
        expect(dsp.dimF <= dsp.nFft / 2 + 1, "$.dsp.dimF exceeds the FFT frequency-bin count.")

        val expectedShape = contract.expectedTensorShape
        expect(
            contract.tensorContract.input.shape == expectedShape,
            "$.tensorContract.input.shape must equal the DSP-derived shape $expectedShape.",
        )
        expect(
            contract.tensorContract.output.shape == expectedShape,
            "$.tensorContract.output.shape must equal the DSP-derived shape $expectedShape.",
        )
        try {
            expect(contract.generationSizeSamples > 0, "DSP-derived generation size must be positive.")
            expectedShape.fold(1) { total, dimension -> Math.multiplyExact(total, dimension) }
        } catch (error: ArithmeticException) {
            throw BenchmarkModelContractException("Contract-derived sample or tensor size overflows Int.", error)
        }
    }

    private fun requireAbsoluteUri(value: String, path: String) {
        val uri = try {
            URI(value)
        } catch (error: Exception) {
            throw BenchmarkModelContractException("$path must be a valid absolute URI.", error)
        }
        expect(uri.isAbsolute, "$path must be an absolute URI.")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private const val SCHEMA_VERSION = 2
    private const val TFLITE_FORMAT = "tflite-flatbuffer"
    private const val FLOAT32 = "float32"
    private const val NHWC = "NHWC"
    private const val PERIODIC_HANN = "periodic-hann"
    private const val RESIDUAL_RULE = "mixture-minus-scaled-model-output"
    private const val PIPELINE_ID = "booming-ss-mdx-stft"
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val GIT_REVISION = Regex("^[0-9a-f]{40}$")
    private val MODEL_ID = Regex("^[a-z0-9_]+$")
    private val CONTRACT_ID = Regex("^[a-z0-9_]+@2$")
}

private class ObjectReader(
    private val value: JsonObject,
    private val path: String,
) {
    fun requireExactKeys(vararg expected: String) {
        val expectedKeys = expected.toSet()
        val missing = expectedKeys - value.values.keys
        val unknown = value.values.keys - expectedKeys
        expect(missing.isEmpty(), "$path is missing required fields: ${missing.sorted().joinToString()}.")
        expect(unknown.isEmpty(), "$path contains unknown fields: ${unknown.sorted().joinToString()}.")
    }

    fun string(name: String): String = required(name).asString(childPath(name))

    fun nonEmptyString(name: String): String = string(name).also {
        expect(it.isNotEmpty(), "${childPath(name)} must not be empty.")
    }

    fun int(name: String): Int = required(name).asInt(childPath(name))

    fun positiveInt(name: String): Int = required(name).asPositiveInt(childPath(name))

    fun positiveLong(name: String): Long = required(name).asPositiveLong(childPath(name))

    fun positiveDouble(name: String): Double = required(name).asPositiveDouble(childPath(name))

    fun obj(name: String): JsonObject = required(name).asObject(childPath(name))

    fun array(name: String): JsonArray = required(name).asArray(childPath(name))

    private fun required(name: String): JsonValue = value.values[name]
        ?: invalid("${childPath(name)} is required.")

    private fun childPath(name: String): String = "$path.$name"
}

private sealed interface JsonValue

private data class JsonObject(val values: LinkedHashMap<String, JsonValue>) : JsonValue

private data class JsonArray(val values: List<JsonValue>) : JsonValue

private data class JsonString(val value: String) : JsonValue

private data class JsonNumber(val value: BigDecimal) : JsonValue

private data class JsonBoolean(val value: Boolean) : JsonValue

private data object JsonNull : JsonValue

private fun JsonValue.asObject(path: String): JsonObject = this as? JsonObject
    ?: invalid("$path must be an object, got ${typeName()}.")

private fun JsonValue.asArray(path: String): JsonArray = this as? JsonArray
    ?: invalid("$path must be an array, got ${typeName()}.")

private fun JsonValue.asString(path: String): String = (this as? JsonString)?.value
    ?: invalid("$path must be a string, got ${typeName()}.")

private fun JsonValue.asInt(path: String): Int {
    val number = (this as? JsonNumber)?.value
        ?: invalid("$path must be an integer, got ${typeName()}.")
    return try {
        number.toBigIntegerExact().intValueExact()
    } catch (error: ArithmeticException) {
        throw BenchmarkModelContractException("$path must be a 32-bit integer.", error)
    }
}

private fun JsonValue.asPositiveInt(path: String): Int = asInt(path).also {
    expect(it > 0, "$path must be positive.")
}

private fun JsonValue.asPositiveLong(path: String): Long {
    val number = (this as? JsonNumber)?.value
        ?: invalid("$path must be an integer, got ${typeName()}.")
    val result = try {
        number.toBigIntegerExact().longValueExact()
    } catch (error: ArithmeticException) {
        throw BenchmarkModelContractException("$path must be a 64-bit integer.", error)
    }
    expect(result > 0L, "$path must be positive.")
    return result
}

private fun JsonValue.asPositiveDouble(path: String): Double {
    val number = (this as? JsonNumber)?.value
        ?: invalid("$path must be a number, got ${typeName()}.")
    expect(number > BigDecimal.ZERO, "$path must be positive.")
    return number.toDouble().also {
        expect(it.isFinite(), "$path must be finite.")
    }
}

private fun JsonValue.typeName(): String = when (this) {
    is JsonObject -> "object"
    is JsonArray -> "array"
    is JsonString -> "string"
    is JsonNumber -> "number"
    is JsonBoolean -> "boolean"
    JsonNull -> "null"
}

private class StrictJsonParser(private val source: String) {
    private var index = 0

    fun parse(): JsonValue {
        skipWhitespace()
        val value = parseValue()
        skipWhitespace()
        if (index != source.length) fail("Unexpected trailing content")
        return value
    }

    private fun parseValue(): JsonValue {
        if (index >= source.length) fail("Expected a JSON value")
        return when (source[index]) {
            '{' -> parseObject()
            '[' -> parseArray()
            '"' -> JsonString(parseString())
            't' -> parseLiteral("true", JsonBoolean(true))
            'f' -> parseLiteral("false", JsonBoolean(false))
            'n' -> parseLiteral("null", JsonNull)
            '-', in '0'..'9' -> parseNumber()
            else -> fail("Unexpected character '${source[index]}'")
        }
    }

    private fun parseObject(): JsonObject {
        consume('{')
        skipWhitespace()
        val values = linkedMapOf<String, JsonValue>()
        if (tryConsume('}')) return JsonObject(values)
        while (true) {
            if (index >= source.length || source[index] != '"') fail("Expected an object field name")
            val name = parseString()
            if (values.containsKey(name)) fail("Duplicate object field '$name'")
            skipWhitespace()
            consume(':')
            skipWhitespace()
            values[name] = parseValue()
            skipWhitespace()
            if (tryConsume('}')) return JsonObject(values)
            consume(',')
            skipWhitespace()
        }
    }

    private fun parseArray(): JsonArray {
        consume('[')
        skipWhitespace()
        val values = mutableListOf<JsonValue>()
        if (tryConsume(']')) return JsonArray(values)
        while (true) {
            values += parseValue()
            skipWhitespace()
            if (tryConsume(']')) return JsonArray(values)
            consume(',')
            skipWhitespace()
        }
    }

    private fun parseString(): String {
        consume('"')
        val result = StringBuilder()
        while (index < source.length) {
            val character = source[index++]
            when {
                character == '"' -> return result.toString()
                character == '\\' -> result.append(parseEscape())
                character.code < 0x20 -> fail("Unescaped control character in string")
                else -> result.append(character)
            }
        }
        fail("Unterminated string")
    }

    private fun parseEscape(): Char {
        if (index >= source.length) fail("Unterminated escape sequence")
        return when (val escaped = source[index++]) {
            '"', '\\', '/' -> escaped
            'b' -> '\b'
            'f' -> '\u000c'
            'n' -> '\n'
            'r' -> '\r'
            't' -> '\t'
            'u' -> parseUnicodeEscape()
            else -> fail("Invalid escape sequence \\$escaped")
        }
    }

    private fun parseUnicodeEscape(): Char {
        if (index + 4 > source.length) fail("Incomplete Unicode escape")
        val digits = source.substring(index, index + 4)
        if (!digits.all { it.isDigit() || it.lowercaseChar() in 'a'..'f' }) {
            fail("Invalid Unicode escape")
        }
        index += 4
        return digits.toInt(16).toChar()
    }

    private fun parseNumber(): JsonNumber {
        val start = index
        if (tryConsume('-') && index >= source.length) fail("Incomplete number")
        if (tryConsume('0')) {
            if (index < source.length && source[index].isDigit()) fail("Leading zero in number")
        } else {
            if (index >= source.length || source[index] !in '1'..'9') fail("Invalid number")
            while (index < source.length && source[index].isDigit()) index++
        }
        if (tryConsume('.')) {
            if (index >= source.length || !source[index].isDigit()) fail("Invalid number fraction")
            while (index < source.length && source[index].isDigit()) index++
        }
        if (index < source.length && source[index] in charArrayOf('e', 'E')) {
            index++
            if (index < source.length && source[index] in charArrayOf('+', '-')) index++
            if (index >= source.length || !source[index].isDigit()) fail("Invalid number exponent")
            while (index < source.length && source[index].isDigit()) index++
        }
        return try {
            JsonNumber(BigDecimal(source.substring(start, index)))
        } catch (error: NumberFormatException) {
            throw BenchmarkModelContractException("Invalid JSON number at character $start.", error)
        }
    }

    private fun <T : JsonValue> parseLiteral(literal: String, value: T): T {
        if (!source.regionMatches(index, literal, 0, literal.length)) fail("Invalid JSON literal")
        index += literal.length
        return value
    }

    private fun consume(expected: Char) {
        if (!tryConsume(expected)) fail("Expected '$expected'")
    }

    private fun tryConsume(expected: Char): Boolean {
        if (index < source.length && source[index] == expected) {
            index++
            return true
        }
        return false
    }

    private fun skipWhitespace() {
        while (index < source.length && source[index] in charArrayOf(' ', '\t', '\r', '\n')) index++
    }

    private fun fail(message: String): Nothing =
        invalid("$message at character $index.")
}

private fun expect(condition: Boolean, message: String) {
    if (!condition) invalid(message)
}

private fun invalid(message: String): Nothing = throw BenchmarkModelContractException(message)
