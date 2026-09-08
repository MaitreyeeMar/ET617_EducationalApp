package org.nplusone.aksharvel.assess

/**
 * The seam that keeps the ASR off the Milestone 1 critical path — and answers the
 * Sarvam question structurally.
 *
 * Every backend gets scored the same way and returns the same three bands, so the UI
 * and the progress store never learn which model is behind it. Swapping IndicConformer
 * for Sarvam Edge (if the SDK opens up), or for a keyword spotter, is one line in the
 * app module.
 *
 * The interface deliberately takes the ON-SCREEN CANDIDATE SET, not just the target.
 * That is not a convenience — it is the contract. Any backend that cannot use it is
 * reduced to transcribe-and-string-compare, which is the design that mislabels
 * accented children. If a vendor cannot accept candidates or expose posteriors, that
 * fact surfaces here at integration time instead of in the field.
 */
interface SpeechScorer {

    enum class Band { ACCEPTED, RETRY, NO_SPEECH, UNAVAILABLE }

    data class Result(
        val band: Band,
        val heardAs: String?,          // which on-screen candidate best matched
        val margin: Float,             // gap to the runner-up; 0 when not applicable
        val weakestAkshara: String?,   // per-akshara GOP, when the backend can provide it
        val backend: String
    ) {
        val accepted get() = band == Band.ACCEPTED
    }

    /** True when a model is present and loaded. False makes the UI hide, not break. */
    val available: Boolean

    /** Human-readable name for the facilitator diagnostics screen. */
    val name: String

    suspend fun score(
        audio: FloatArray,
        sampleRate: Int,
        target: String,
        onScreen: List<String>,
        aksharas: List<String>
    ): Result
}

/**
 * Milestone 1 backend. Captures real audio, applies the real silence gate, and returns
 * UNAVAILABLE rather than a fabricated verdict.
 *
 * This is the honest thing to demo on 6 Sep: the capture path, the permission flow, the
 * live level meter and the retry loop are all genuinely working and are most of the
 * integration risk. Pretending to grade pronunciation with a random number would make
 * the demo look better and the schedule worse, because the first real threshold
 * calibration would then be mistaken for a regression.
 *
 * Mode 2 falls back to reference-audio-then-record-and-replay while this is active, so
 * a child on a Milestone 1 build still gets a complete, useful activity.
 */
class StubSpeechScorer : SpeechScorer {
    override val available = false
    override val name = "stub (no model bundled)"

    override suspend fun score(
        audio: FloatArray,
        sampleRate: Int,
        target: String,
        onScreen: List<String>,
        aksharas: List<String>
    ): SpeechScorer.Result {
        var peak = 0f
        for (v in audio) { val a = kotlin.math.abs(v); if (a > peak) peak = a }
        // 0.02 ~= -34 dBFS. Recalibrate in a real classroom, not at a desk.
        val band = if (peak < 0.02f) SpeechScorer.Band.NO_SPEECH else SpeechScorer.Band.UNAVAILABLE
        return SpeechScorer.Result(band, null, 0f, null, name)
    }
}
