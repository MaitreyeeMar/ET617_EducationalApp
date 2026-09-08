package org.nplusone.aksharvel.ui.speaking

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.nplusone.aksharvel.assess.SpeechScorer
import org.nplusone.aksharvel.data.AksharvelDao
import org.nplusone.aksharvel.data.Curriculum
import org.nplusone.aksharvel.data.Mode
import org.nplusone.aksharvel.data.record

/**
 * SpeakingViewModel — Mode 2 (बोला) state machine.
 *
 * Owns the full speaking-exercise lifecycle:
 *   IDLE  →  RECORDING  →  SCORING  →  RESULT  →  (next glyph or finished)
 *
 * The VM talks to:
 *   - [SpeechScorer]       (Yugratna's layer) for audio capture + pronunciation verdict
 *   - [PhonemeHighlighter] (this module)      to map the verdict to per-akshara colours
 *   - [AksharvelDao]       (Maitreyee's DB)   to persist the attempt
 *
 * The Fragment only observes [uiState] — it never calls the scorer or the DB directly.
 * This keeps the UI layer thin and the business logic testable without Android.
 *
 * Threading: all coroutines run on Dispatchers.Default (inside SpeechScorer.score)
 * or Dispatchers.IO (DB writes). StateFlow emissions are collected on the main thread
 * by the Fragment.
 */
class SpeakingViewModel(
    private val scorer:    SpeechScorer,
    private val dao:       AksharvelDao,
    private val learnerId: String,
    private val glyphs:    List<Curriculum.Item>   // the session's glyph list, in order
) : ViewModel() {

    // ---------------------------------------------------------------- UI state

    sealed class UiState {
        /** Waiting for the child to tap the mic button. Shows the glyph + reference audio button. */
        data class Idle(
            val item: Curriculum.Item,
            val attemptNo: Int,
            val aksharaStates: List<AksharaState>    // all PENDING on first load
        ) : UiState()

        /** Mic is open. Show live level meter; hide score UI. */
        data class Recording(
            val item: Curriculum.Item,
            val level: Float                         // 0..1, updated ~10 Hz
        ) : UiState()

        /** Model is running. Show a brief "thinking" indicator. */
        data class Scoring(val item: Curriculum.Item) : UiState()

        /**
         * Verdict is ready. Show per-akshara green/amber/grey row, overall icon,
         * and the aural hint string (null when fully correct).
         */
        data class Result(
            val item:         Curriculum.Item,
            val aksharaStates: List<AksharaState>,
            val band:         SpeechScorer.Band,
            val auralHint:    String?,
            val attemptNo:    Int,
            val canAdvance:   Boolean                // false while the last attempt still plays
        ) : UiState()

        /** All glyphs in this session are done. */
        object Finished : UiState()
    }

    private val _uiState = MutableStateFlow<UiState>(buildIdle(index = 0, attemptNo = 1))
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var glyphIndex = 0
    private var attemptNo  = 1
    private var recordJob: Job? = null

    // ---------------------------------------------------------------- public actions

    /** Called when the child taps the microphone button. */
    fun onMicTap() {
        val current = _uiState.value
        if (current !is UiState.Idle) return
        val item = current.item

        recordJob = viewModelScope.launch {
            var stopped = false
            val audio = scorer.let { s ->
                // SpeechScorer.record is blocking; run it in the coroutine
                // and update the level meter via onLevel callback.
                // NOTE: PronunciationChecker.record is not a suspend fun —
                // wrap it appropriately when wiring to the real implementation.
                null  // replaced by real capture in integration; see note below
            }
            // ── Milestone 1 behaviour ──────────────────────────────────────────
            // StubSpeechScorer.available == false, so we jump straight to scoring
            // with a zero-length buffer to get a UNAVAILABLE result. The child
            // still sees the full feedback flow (colour row, hint) with grey aksharas,
            // and the facilitator sees "stub (no model bundled)" in diagnostics.
            // This is intentional — see SpeechScorer.kt header.
            val dummyAudio = FloatArray(0)
            _uiState.update { UiState.Scoring(item) }

            val result = scorer.score(
                audio      = dummyAudio,
                sampleRate = 16_000,
                target     = item.glyph,
                onScreen   = Curriculum.optionsFor(item.glyph),
                aksharas   = item.aksharas
            )
            onScoringComplete(item, result)
        }
    }

    /** Called when the child taps the stop button (early termination). */
    fun onStopRecording() {
        recordJob?.cancel()
        val current = _uiState.value
        if (current is UiState.Recording) {
            _uiState.update { UiState.Idle(current.item, attemptNo, pendingStates(current.item)) }
        }
    }

    /** Called when the "Next" button is tapped after a result is shown. */
    fun onAdvance() {
        glyphIndex++
        attemptNo = 1
        if (glyphIndex >= glyphs.size) {
            _uiState.update { UiState.Finished }
        } else {
            _uiState.update { buildIdle(glyphIndex, attemptNo) }
        }
    }

    /** Called when the "Try again" button is tapped (max [Curriculum.MAX_ATTEMPTS_BEFORE_ACCEPT]). */
    fun onRetry() {
        if (attemptNo >= Curriculum.MAX_ATTEMPTS_BEFORE_ACCEPT) {
            onAdvance()   // forced advance — never let a failing attempt block a child
            return
        }
        attemptNo++
        _uiState.update { buildIdle(glyphIndex, attemptNo) }
    }

    // ---------------------------------------------------------------- internal

    private fun onScoringComplete(item: Curriculum.Item, result: SpeechScorer.Result) {
        // Choose the right highlight strategy based on what the scorer can provide.
        val states = if (result.gopScores != null) {
            PhonemeHighlighter.highlight(result, item.aksharas)
        } else {
            PhonemeHighlighter.highlightBandOnly(result, item.aksharas)
        }
        val hint = PhonemeHighlighter.auralHint(states)

        _uiState.update {
            UiState.Result(
                item          = item,
                aksharaStates = states,
                band          = result.band,
                auralHint     = hint,
                attemptNo     = attemptNo,
                canAdvance    = true
            )
        }

        // Persist the attempt asynchronously — DB write must not block the UI.
        viewModelScope.launch {
            dao.record(
                learnerId = learnerId,
                glyph     = item.glyph,
                mode      = Mode.SPEAK,
                correct   = result.accepted,
                detail    = result.band.name,
                attemptNo = attemptNo
            )
        }
    }

    private fun buildIdle(index: Int, attemptNo: Int): UiState.Idle {
        val item = glyphs[index]
        return UiState.Idle(item, attemptNo, pendingStates(item))
    }

    private fun pendingStates(item: Curriculum.Item) =
        item.aksharas.map { AksharaState(it, PhonemeHighlighter.State.PENDING) }
}

// ── Extension to bridge SpeechScorer.Result to the GOP array ──────────────────────────
// SpeechScorer.Result (Yugratna's interface) stores gopScores as FloatArray?.
// This extension is here so PhonemeHighlighter does not need to know the concrete type.
private val SpeechScorer.Result.gopScores: FloatArray?
    get() = (this as? org.nplusone.aksharvel.speech.PronunciationChecker.Verdict)?.gop
        ?: run {
            // StubSpeechScorer and other backends that don't provide GOP return null.
            null
        }
