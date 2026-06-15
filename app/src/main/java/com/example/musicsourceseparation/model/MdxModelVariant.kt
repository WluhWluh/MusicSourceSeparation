package com.example.musicsourceseparation.model

enum class MdxModelVariant(
    val displayName: String,
    val fileName: String,
    val outputTag: String,
) {
    INST_MAIN(
        displayName = "UVR-MDX-NET Inst Main",
        fileName = "UVR-MDX-NET-Inst_Main.onnx",
        outputTag = "inst_main",
    ),
    MDXNET_9482(
        displayName = "UVR MDXNET 9482",
        fileName = "UVR_MDXNET_9482.onnx",
        outputTag = "mdxnet_9482",
    );

    override fun toString(): String = displayName
}
