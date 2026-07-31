package com.example.musicsourceseparation.benchmark.applive

import org.junit.Assert.assertEquals
import org.junit.Test

class AppLiveRunOriginTest {
    @Test
    fun missingPreferenceDefaultsToBrowserStack() {
        assertEquals(AppLiveRunOrigin.BROWSERSTACK, AppLiveRunOrigin.from(null))
        assertEquals(AppLiveRunOrigin.BROWSERSTACK, AppLiveRunOrigin.from("unknown"))
    }

    @Test
    fun explicitLocalPreferenceRemainsAvailable() {
        assertEquals(AppLiveRunOrigin.LOCAL, AppLiveRunOrigin.from("local"))
    }
}
