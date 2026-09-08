# Speaking Mode — Phoneme Feedback Module

**Author:** Vedika Borse  
**Thread:** Speaking Mode (बोला) — visual feedback layer  
**Meeting ref:** MoM 04 · 02/09/2026

---

## What this module does

This folder bridges Yugratna's ASR pipeline (`PronunciationChecker.kt` / `SpeechScorer`) and
the child-facing screen. It takes the per-akshara GOP scores the scorer produces and maps them
to the green / amber / grey colour row the child sees.

```
PronunciationChecker.kt  →  SpeechScorer.Result
        (Yugratna)                    │
                                      ▼
                           PhonemeHighlighter.kt     ← this module
                                      │
                         List<AksharaState> (colour states)
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                  AksharaRowView.kt        SpeakingViewModel.kt
                  (draws the chips)        (drives the screen state machine)
```

---

## Files

| File | Purpose |
|------|---------|
| `PhonemeHighlighter.kt` | Maps `SpeechScorer.Result` → `List<AksharaState>`. Owns the GOP threshold and the aural hint string. |
| `AksharaState.kt` | *(defined inside PhonemeHighlighter.kt)* Data class holding one akshara's display state and raw GOP score. |
| `AksharaRowView.kt` | Custom `View` that draws the phoneme chip row with animation on the FOCUS chip. |
| `SpeakingViewModel.kt` | ViewModel for the Speaking Mode screen. Orchestrates record → score → highlight → persist. Talks to Yugratna's `SpeechScorer` and Maitreyee's `AksharvelDao`. |
| `validate_gop.py` | Calibration harness for `CORRECT_THRESHOLD`. Run against labelled recordings to verify the threshold is data-driven, not guessed. Mirrors the role of `stroke_validate.py`. |
| `colors_aksharvel.xml` | Android colour resources for the chip states. Merge into `res/values/colors.xml`. |

---

## Design decisions (from MoM §4 and §5)

### 1. Per-phoneme, not per-word
The client explicitly approved phoneme-level green colouring (the *bhakri* example in §5).
`PhonemeHighlighter` maps each akshara individually — a child who says the first sound right
but gets the second wrong sees one green chip and one amber chip, not a binary pass/fail icon.

### 2. No text labels on child-facing UI
Per MoM §4 *Iconography over Text* requirement, `AksharaRowView` uses only colour and the
akshara glyph itself. No English or Marathi instruction text appears in the chip row.

### 3. FOCUS state = weakest akshara
`SpeechScorer.Result.weakestAkshara` is surfaced as a distinct state (animated amber border)
rather than just another WEAK chip. This lets the facilitator see at a glance which sound to
coach the child on next, without having to interpret the raw GOP numbers.

### 4. Works in Milestone 1 with `StubSpeechScorer`
When `SpeechScorer.available == false` (the stub), `PhonemeHighlighter.highlightBandOnly()`
is used instead — all chips go grey on any non-ACCEPTED result. The ViewModel still runs the
full capture path so the mic permission, level meter, and retry loop are exercised in demos.

### 5. GOP threshold is explicit and calibratable
`CORRECT_THRESHOLD = -0.50f` is a named constant with a comment explaining its origin.
`validate_gop.py` is the tool to recalibrate it once child-speech data is available (MoM §3).
The threshold must NOT be buried in the model inference code.

---

## Integration checklist

- [ ] Add `AksharaRowView` to `fragment_speaking.xml`
- [ ] Observe `SpeakingViewModel.uiState` in `SpeakingFragment`, call `aksharaRow.setStates()`
- [ ] Merge `colors_aksharvel.xml` into `res/values/colors.xml`
- [ ] Wire `SpeakingViewModel` factory with the real `SpeechScorer` implementation once Yugratna's APK test is done
- [ ] Run `validate_gop.py` against adult-baseline recordings and update `CORRECT_THRESHOLD` if the suggested value differs by more than 0.1

---

## Running `validate_gop.py`

```bash
pip install onnxruntime scipy matplotlib numpy
python validate_gop.py --recordings data/adult_baseline/ \
                       --tokens assets/asr/tokens.txt \
                       --model  assets/asr/indicconformer_mr_ctc_int8.onnx
```

Without recordings, the script still runs its selftest of the GOP maths:

```bash
python validate_gop.py
# SELFTEST PASSED
```

The selftest validates the Python GOP implementation against a hand-crafted 5-frame example.
Because `validate_gop.py` and `PronunciationChecker.kt` are ports of the same algorithm,
a passing selftest is evidence that the Kotlin is also correct.
