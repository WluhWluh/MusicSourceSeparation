package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class DerivedLiteRtCandidateContractLoaderTest {
    @Test
    fun loadsGatherReshapeDerivativeAndBindsBaseIdentity() {
        val loaded = DerivedLiteRtCandidateContractLoader.load(manifest(), base())
        val contract = loaded.contract

        assertEquals(1, contract.contractSchemaVersion)
        assertEquals("derived-litert-model-candidate", contract.contractKind)
        assertEquals("bandbuddy_htdemucs_6s_core_gather_reshape_v1_0_0@1", contract.contractId)
        assertEquals("bandbuddy_htdemucs_6s_core_v1_0_0@4", contract.base.contractId)
        assertEquals(117_784_760L, contract.base.artifactByteSize)
        assertEquals(117_784_760L, contract.artifact.byteSize)
        assertEquals(
            "ce971b195c20f1233c407b145ce8123f4d4b0834c6d946f0786504b999fb26a6",
            contract.artifact.sha256,
        )
        assertEquals("identity-gather-nd-to-reshape-v1", contract.rewrite.ruleId)
        assertEquals(
            "514bd175c57bd461b80cffbab8c812a8de13daa70b0cdbc816184bf8dfdf66b1",
            contract.rewrite.toolSha256,
        )
        assertEquals(132, contract.rewrite.rewrittenOperatorCount)
        assertEquals(14, contract.rewrite.rewrittenShapeTensorCount)
        assertEquals(0, contract.flatBufferDelta.gatherNdCount)
        assertEquals(132, contract.flatBufferDelta.reshapeDelta)
        assertEquals(4_332, loaded.base.contract.flatBuffer.tensorCount)
    }

    @Test
    fun rejectsManifestThatDoesNotBindBaseArtifact() {
        val source = manifest().readText(Charsets.UTF_8)
        val mutated = source.replaceFirst(
            "a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d",
            "0".repeat(64),
        )
        assertThrows(BenchmarkModelContractException::class.java) {
            DerivedLiteRtCandidateContractLoader.parse(mutated)
        }
    }

    @Test
    fun rejectsDerivativeThatClaimsPreservedBytesWithDifferentSize() {
        val source = manifest().readText(Charsets.UTF_8)
        val pattern = Regex("""("artifact"\s*:\s*\{[\s\S]*?"byteSize"\s*:\s*)117784760""")
        val match = requireNotNull(pattern.find(source))
        val mutated = source.replaceRange(match.range, "${match.groupValues[1]}117784759")
        assertThrows(BenchmarkModelContractException::class.java) {
            DerivedLiteRtCandidateContractLoader.parse(mutated)
        }
    }

    @Test
    fun manifestPinsTheCheckedInRewriteTool() {
        val contract = DerivedLiteRtCandidateContractLoader.parse(manifest().readText(Charsets.UTF_8))
        val tool = repositoryFile(contract.rewrite.toolPath)
        val actualSha256 = MessageDigest.getInstance("SHA-256")
            .digest(tool.readBytes())
            .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
        assertEquals(contract.rewrite.toolSha256, actualSha256)
    }

    private fun manifest(): File = asset("bandbuddy_htdemucs_6s_core_gather_reshape_v1_0_0.json")

    private fun base(): File = asset("bandbuddy_htdemucs_6s_core_v1_0_0.json")

    private fun asset(name: String): File {
        val relative = "src/main/assets/benchmark-contracts/$name"
        return sequenceOf(File(relative), File("app/$relative"))
            .firstOrNull(File::isFile)
            ?: error("Could not locate contract asset $name")
    }

    private fun repositoryFile(relativePath: String): File = sequenceOf(
        File(relativePath),
        File("../$relativePath"),
    ).firstOrNull(File::isFile)
        ?: error("Could not locate repository file $relativePath")
}
