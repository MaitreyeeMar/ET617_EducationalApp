package org.nplusone.aksharvel.ui.speaking

import org.nplusone.aksharvel.assess.SpeechScorer

/**
 * PhonemeHighlighter — Speaking Mode (बोला) visual feedback layer.
 *
 * Sits between Yugratna's SpeechScorer.Result and the screen. Takes the per-akshara
 * GOP (Goodness of Pronunciation) scores from the Result and maps each akshara to one
 * of three display states: CORRECT (green), WEAK (amber), or UNSCORED (grey).
 *
 * Why a separate class:
 *   The SpeechScorer contract deliberately says nothing about colour or UI. This class
 *   owns that mapping so it can be tuned independently — the thresholds below were
 *   chosen so that a child who says a phoneme recognisably but with a regional accent
 *   sees green, not amber. That calibration must NOT live inside PronunciationChecker.kt
 *   because it is a pedagogical decision, not an acoustic one.
 *
 * Usage (from SpeakingViewModel or SpeakingFragment):
 *   val states = PhonemeHighlighter.highlight(result, curriculum.aksharas)
 *   // states[i] corresponds to aksharas[i], in order.
 *
 * Thread-safety: stateless — every call returns a new list. Safe to call on any thread.
 */
object PhonemeHighlighter {

    /**
     * Visual state for a single akshara slot in the speaking exercise.
     *
     *  PENDING   — child has not spoken yet (initial state)
     *  CORRECT   — green:  GOP above [CORRECT_THRESHOLD]
     *  WEAK      — amber:  GOP below threshold but not the single weakest
     *  FOCUS     — amber with underline pulse: the [weakestAkshara] the scorer named
     *  UNSCORED  — grey:   model returned NEG_INF (silence or alignment failure)
     */
    enum class State { PENDING, CORRECT, WEAK, FOCUS, UNSCORED }

    /**
     * GOP is log-scaled: 0 = perfect, more negative = worse.
     * -0.5 was chosen by running the stub scorer against 20 adult Marathi recordings
     * (see validate_gop.py).  Recalibrate once child-speech data is available.
     */
    private const val CORRECT_THRESHOLD = -0.50f
    private const val NEG_INF_GUARD     = -1e20f   // matches PronunciationChecker.NEG_INF

    /**
     * Map a SpeechScorer.Result onto per-akshara display states.
     *
     * @param result      the verdict from SpeechScorer.score()
     * @param aksharas    the ordered akshara list for the current glyph (from Curriculum)
     * @return            one State per akshara, same length and order as [aksharas]
     */
    fun highlight(result: SpeechScorer.Result, aksharas: List<String>): List<AksharaState> {
        if (result.band == SpeechScorer.Band.NO_SPEECH ||
            result.band == SpeechScorer.Band.UNAVAILABLE) {
            return aksharas.map { AksharaState(it, State.PENDING) }
        }

        val gop: FloatArray? = result.gopScores

        return aksharas.mapIndexed { index, akshara ->
            val state = when {
                gop == null || index >= gop.size   -> State.UNSCORED
                gop[index] <= NEG_INF_GUARD        -> State.UNSCORED
                akshara == result.weakestAkshara   -> State.FOCUS
                gop[index] >= CORRECT_THRESHOLD    -> State.CORRECT
                else                               -> State.WEAK
            }
            AksharaState(akshara, state, gop?.getOrNull(index))
        }
    }

    /**
     * Overload for when you only have the overall band and weakest akshara, but no
     * per-akshara GOP (e.g. StubSpeechScorer in Milestone 1).
     * All aksharas show CORRECT on ACCEPTED, PENDING on anything else.
     */
    fun highlightBandOnly(result: SpeechScorer.Result, aksharas: List<String>): List<AksharaState> {
        val state = if (result.band == SpeechScorer.Band.ACCEPTED) State.CORRECT else State.PENDING
        return aksharas.map { AksharaState(it, state, gopScore = null) }
    }

    /**
     * Returns the Marathi hint string the audio engine should speak aloud when the
     * child's attempt ends with at least one FOCUS or WEAK akshara.
     * Returns null when the attempt was fully correct (no hint needed).
     *
     * The hint names the weakest akshara directly so the facilitator can echo it.
     * Deliberately short — this runs on-device TTS, which is slow on budget SoCs.
     */
    fun auralHint(states: List<AksharaState>): String? {
        val focus = states.firstOrNull { it.state == State.FOCUS }
            ?: states.firstOrNull { it.state == State.WEAK }
            ?: return null
        // "{akshara} पुन्हा सांग" = "Say {akshara} again"
        return "${focus.akshara} पुन्हा सांग"
    }
}

/**
 * Holds the display state for one akshara slot, plus the raw GOP score for
 * the facilitator diagnostics screen (HLD §5.2 / UC-04).
 */
data class AksharaState(
    val akshara: String,
    val state:   PhonemeHighlighter.State,
    val gopScore: Float? = null        // null when the backend cannot provide it
) {
    /** Convenience: is this akshara considered fully correct? */
    val isCorrect get() = state == PhonemeHighlighter.State.CORRECT
}
