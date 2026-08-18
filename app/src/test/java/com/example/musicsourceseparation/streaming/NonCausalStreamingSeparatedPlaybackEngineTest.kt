package com.example.musicsourceseparation.streaming

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NonCausalStreamingSeparatedPlaybackEngineTest {
    private val model = StreamingModelConfig("tfc-tdf-test")

    @Test
    fun selectsDryImmediatelyAndWetAfterBackgroundWindowCompletes() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val factory = RecordingFactory(processStarted = started, processGate = release)
        val reader = FakeReader(frameCount = model.usefulSamples * 3 + 777)
        val engine = NonCausalStreamingSeparatedPlaybackEngine(
            reader = reader,
            sessionFactory = factory,
        )
        try {
            engine.start(model = model, accelerator = StreamingAccelerator.CPU)
            assertTrue(started.await(2, TimeUnit.SECONDS))

            val dry = FloatArray(1_024 * 2) { 0.5f }
            val output = FloatArray(dry.size)
            val startNanos = System.nanoTime()
            val initialMode = engine.selectBlock(0, dry, output)
            val elapsedMillis = (System.nanoTime() - startNanos) / 1_000_000

            assertEquals(StreamingPlaybackMode.DRY, initialMode)
            assertTrue("playback selection took ${elapsedMillis} ms", elapsedMillis < 100)
            assertEquals(dry.toList(), output.toList())

            release.countDown()
            assertTrue(await { engine.snapshot().publishedWindowCount > 0 })
            val wetMode = engine.selectBlock(1_024, dry, output)
            assertEquals(StreamingPlaybackMode.WET, wetMode)
            assertEquals(reader.samples[1_024 * 2] * 0.25f, output[0], 1e-6f)
        } finally {
            release.countDown()
            engine.close()
        }
    }

    @Test
    fun lateWetResultCannotBackfillAlreadyEmittedDryBlocks() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val factory = RecordingFactory(processStarted = started, processGate = release)
        val reader = FakeReader(frameCount = model.usefulSamples * 2)
        val engine = NonCausalStreamingSeparatedPlaybackEngine(reader, factory)
        try {
            engine.start(model = model)
            assertTrue(started.await(2, TimeUnit.SECONDS))
            val dry = FloatArray(1_024 * 2) { 0.4f }
            val output = FloatArray(dry.size)
            assertEquals(StreamingPlaybackMode.DRY, engine.selectBlock(0, dry, output))
            assertEquals(StreamingPlaybackMode.DRY, engine.selectBlock(1_024, dry, output))

            release.countDown()
            assertTrue(await { engine.snapshot().publishedWindowCount > 0 })
            assertEquals(
                "a late result must not replace the already emitted range",
                StreamingPlaybackMode.DRY,
                engine.selectBlock(0, dry, output),
            )
        } finally {
            release.countDown()
            engine.close()
        }
    }

    @Test
    fun seekInvalidatesOldGenerationBeforePublishingItsResult() {
        val factory = RecordingFactory(processDelayMillis = 250, ignoreInterrupt = true)
        val reader = FakeReader(frameCount = model.usefulSamples * 4)
        val engine = NonCausalStreamingSeparatedPlaybackEngine(reader, factory)
        try {
            engine.start(model = model)
            assertTrue(await { factory.processCount.get() > 0 })
            engine.seek(model.usefulSamples.toLong() + 123)

            assertTrue(await { engine.snapshot().publishedWindowCount > 0 })
            val snapshot = engine.snapshot()
            assertEquals(2, snapshot.epoch)
            assertTrue(snapshot.discardedEpochOutputCount > 0)
            assertEquals(1, factory.openCount.get())

            val publishedAfterFirstSeek = snapshot.publishedWindowCount
            engine.seek(model.usefulSamples.toLong() * 2 + 123)
            assertTrue(await { engine.snapshot().publishedWindowCount > publishedAfterFirstSeek })
            assertEquals(3, engine.snapshot().epoch)
            assertEquals(1, factory.openCount.get())
        } finally {
            engine.close()
        }
    }

    @Test
    fun modelOrAcceleratorSwitchClosesAndReplacesReusableSession() {
        val factory = RecordingFactory()
        val reader = FakeReader(frameCount = model.usefulSamples * 5)
        val engine = NonCausalStreamingSeparatedPlaybackEngine(reader, factory)
        val replacement = model.copy(modelId = "tfc-tdf-replacement")
        try {
            engine.start(model = model, accelerator = StreamingAccelerator.GPU)
            assertTrue(await { factory.processCount.get() > 0 })

            engine.switchModel(replacement, StreamingAccelerator.GPU)
            assertTrue(await { factory.openCount.get() == 2 })
            assertEquals(1, factory.closeCount.get())
            assertEquals(replacement, factory.lastModel)

            engine.switchModel(replacement, StreamingAccelerator.CPU)
            assertTrue(await { factory.openCount.get() == 3 })
            assertEquals(2, factory.closeCount.get())
            assertEquals(StreamingAccelerator.CPU, factory.lastAccelerator)
        } finally {
            engine.close()
        }
        assertEquals(3, factory.closeCount.get())
    }

    @Test
    fun reusesOneCpuOrGpuSessionAcrossWindowsAndBoundsInputMemory() {
        val factory = RecordingFactory()
        val reader = FakeReader(frameCount = model.usefulSamples * 5)
        val engine = NonCausalStreamingSeparatedPlaybackEngine(
            reader = reader,
            sessionFactory = factory,
            maxInputWindows = 3,
        )
        try {
            engine.start(model = model, accelerator = StreamingAccelerator.GPU)
            assertTrue(await { factory.processCount.get() >= 2 })
            val snapshot = engine.snapshot()
            assertEquals(1, factory.openCount.get())
            assertEquals(StreamingAccelerator.GPU, factory.lastAccelerator)
            assertTrue(snapshot.maxInputRingSamples <= 4L * model.usefulSamples)
            assertTrue(snapshot.wetWindowCount <= 4)
        } finally {
            engine.close()
        }
    }

    @Test
    fun selectsWetAcrossAdjacentWindowBoundaryWithoutCrossfade() {
        val factory = RecordingFactory()
        val reader = FakeReader(frameCount = model.usefulSamples * 3 + 5)
        val engine = NonCausalStreamingSeparatedPlaybackEngine(reader, factory)
        try {
            engine.start(model = model)
            assertTrue(await { engine.snapshot().publishedWindowCount >= 2 })
            val position = model.usefulSamples.toLong() - 500
            val dry = FloatArray(1_000 * 2) { -0.75f }
            val output = FloatArray(dry.size)
            val mode = engine.selectBlock(position, dry, output)

            assertEquals(StreamingPlaybackMode.WET, mode)
            assertEquals(reader.samples[position.toInt() * 2] * 0.25f, output[0], 1e-6f)
            assertEquals(
                reader.samples[(position.toInt() + 500) * 2] * 0.25f,
                output[1_000],
                1e-6f,
            )
        } finally {
            engine.close()
        }
    }

    private fun await(timeoutMillis: Long = 3_000, condition: () -> Boolean): Boolean {
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis)
        while (System.nanoTime() < deadline) {
            if (condition()) return true
            Thread.sleep(5)
        }
        return condition()
    }

    private class FakeReader(frameCount: Int) : StreamingAudioReader {
        val samples = FloatArray(frameCount * 2) { index ->
            ((index % 2_000) - 1_000) / 1_000f
        }

        override val sampleRate: Int = 44_100
        override val channelCount: Int = 2
        override val frameCount: Long = frameCount.toLong()

        override fun read(startSample: Long, frameCount: Int): FloatArray {
            val start = startSample.toInt() * 2
            return samples.copyOfRange(start, start + frameCount * 2)
        }

        override fun close() = Unit
    }

    private class RecordingFactory(
        private val processDelayMillis: Long = 0,
        private val ignoreInterrupt: Boolean = false,
        private val processStarted: CountDownLatch? = null,
        private val processGate: CountDownLatch? = null,
    ) : StreamingInferenceSessionFactory {
        val openCount = AtomicInteger(0)
        val closeCount = AtomicInteger(0)
        val processCount = AtomicInteger(0)
        @Volatile
        var lastModel: StreamingModelConfig? = null
        @Volatile
        var lastAccelerator: StreamingAccelerator? = null

        override fun open(
            model: StreamingModelConfig,
            accelerator: StreamingAccelerator,
        ): StreamingInferenceSession {
            openCount.incrementAndGet()
            lastModel = model
            lastAccelerator = accelerator
            return object : StreamingInferenceSession {
                override fun process(inputPcm: FloatArray, actualSamples: Int): FloatArray {
                    processCount.incrementAndGet()
                    processStarted?.countDown()
                    processGate?.await()
                    if (processDelayMillis > 0) {
                        val deadline = System.nanoTime() +
                            TimeUnit.MILLISECONDS.toNanos(processDelayMillis)
                        while (System.nanoTime() < deadline) {
                            try {
                                Thread.sleep(5)
                            } catch (interrupted: InterruptedException) {
                                if (!ignoreInterrupt) throw interrupted
                            }
                        }
                    }
                    val start = model.trimSamples * 2
                    return inputPcm.copyOfRange(start, start + actualSamples * 2)
                        .also { values ->
                            values.indices.forEach { index -> values[index] *= 0.25f }
                        }
                }

                override fun close() {
                    closeCount.incrementAndGet()
                }
            }
        }
    }
}
