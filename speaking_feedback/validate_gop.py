"""
validate_gop.py  —  Calibration harness for PhonemeHighlighter.CORRECT_THRESHOLD.

Mirrors the role that stroke_validate.py plays for the handwriting classifier:
a standalone Python script that can be run without Android to verify that the
threshold used in Kotlin matches what the data actually shows.

What it does
------------
1. Loads a set of labelled test recordings (adult Marathi baseline, as per MoM §3).
2. Runs the CTC forward pass + GOP computation in pure Python (same maths as
   PronunciationChecker.kt — see inline comments for the correspondence).
3. Plots GOP score distributions for correctly-pronounced vs incorrectly-pronounced
   aksharas so the threshold can be set from data, not from intuition.
4. Prints a confusion matrix and suggests a threshold that maximises F1 on the test set.

Running
-------
  python validate_gop.py --recordings data/adult_baseline/ --tokens assets/asr/tokens.txt

The script expects recordings/ to contain:
  <glyph>_<label>_<n>.wav    e.g.  ka_correct_01.wav   or   ka_wrong_03.wav
  where <label> is "correct" or "wrong".

Dependencies: numpy, scipy, onnxruntime, matplotlib (all available via pip).
The ONNX model path defaults to assets/asr/indicconformer_mr_ctc_int8.onnx;
override with --model.
"""

import argparse
import math
import pathlib
import re
import sys
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt
    FULL_MODE = True
except ImportError:
    FULL_MODE = False
    print("[warn] onnxruntime / scipy / matplotlib not installed — running selftest only.\n")


# ───────────────────────────────────────────────────────── GOP in pure Python
# This is the reference implementation that PronunciationChecker.kt was ported from.
# If you change the Kotlin, update this too; they must stay in sync.

NEG_INF = -1e30

def log_add(a: float, b: float) -> float:
    if a <= NEG_INF / 2: return b
    if b <= NEG_INF / 2: return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))

def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - math.log(np.exp(x).sum())

def ctc_log_prob(log_probs: np.ndarray, labels: list[int], blank: int = 0) -> float:
    """Exact log P(labels | audio) under CTC. Port of PronunciationChecker.ctcLogProb."""
    T, V = log_probs.shape
    if T < len(labels):
        return NEG_INF
    ext = []
    for tok in labels:
        ext.append(blank)
        ext.append(tok)
    ext.append(blank)
    S = len(ext)
    a = [NEG_INF] * S
    a[0] = log_probs[0][ext[0]]
    if S > 1:
        a[1] = log_probs[0][ext[1]]
    for t in range(1, T):
        nxt = [NEG_INF] * S
        lo = max(0, S - 2 * (T - t))
        for i in range(lo, S):
            v = a[i]
            if i > 0: v = log_add(v, a[i - 1])
            if i > 1 and ext[i] != blank and ext[i] != ext[i - 2]:
                v = log_add(v, a[i - 2])
            nxt[i] = v + log_probs[t][ext[i]]
        a = nxt
    return log_add(a[-1], a[-2] if S > 1 else NEG_INF)

def gop(log_probs: np.ndarray, labels: list[int], blank: int = 0) -> list[float]:
    """
    Per-token GOP via Viterbi path. Port of PronunciationChecker.gop().
    Returns one score per label; 0 is perfect, more negative is worse.
    """
    T, _ = log_probs.shape
    if not labels or T < len(labels):
        return [NEG_INF] * len(labels)

    ext = []
    for tok in labels:
        ext.append(blank)
        ext.append(tok)
    ext.append(blank)
    S = len(ext)

    d   = [[NEG_INF] * S for _ in range(T)]
    bp  = [[0]        * S for _ in range(T)]
    d[0][0] = log_probs[0][ext[0]]
    if S > 1:
        d[0][1] = log_probs[0][ext[1]]

    for t in range(1, T):
        for i in range(S):
            best_v, best_i = d[t-1][i], i
            if i > 0 and d[t-1][i-1] > best_v:
                best_v, best_i = d[t-1][i-1], i-1
            if i > 1 and ext[i] != blank and ext[i] != ext[i-2] and d[t-1][i-2] > best_v:
                best_v, best_i = d[t-1][i-2], i-2
            d[t][i] = best_v + log_probs[t][ext[i]]
            bp[t][i] = best_i

    cur = S-1 if (S == 1 or d[T-1][S-1] >= d[T-1][S-2]) else S-2
    path = [0] * T
    for t in range(T-1, -1, -1):
        path[t] = cur
        cur = bp[t][cur]

    total  = [0.0] * len(labels)
    counts = [0]   * len(labels)
    for t in range(T):
        i = path[t]
        if ext[i] == blank:
            continue
        li = (i - 1) // 2
        total[li]  += log_probs[t][ext[i]] - log_probs[t].max()
        counts[li] += 1

    return [total[j] / counts[j] if counts[j] > 0 else NEG_INF for j in range(len(labels))]


# ───────────────────────────────────────────────────────── selftest (always runs)

def selftest() -> bool:
    """
    Verify the GOP implementation against a hand-crafted 2-token, 5-frame example.
    If this fails, the Kotlin port will also be wrong.
    """
    ok = True
    V = 4   # vocab: 0=blank, 1=a, 2=b, 3=noise
    T = 5

    # Perfect evidence: frame 0-1 = token 1, frame 2 = blank, frame 3-4 = token 2.
    lp = np.full((T, V), -10.0)
    lp[0, 1] = 0.0;  lp[1, 1] = 0.0
    lp[2, 0] = 0.0
    lp[3, 2] = 0.0;  lp[4, 2] = 0.0
    lp = np.array([log_softmax(row) for row in lp])

    labels = [1, 2]
    scores = gop(lp, labels)
    # Both tokens should score near 0 (perfect)
    for i, s in enumerate(scores):
        good = s > -0.2
        ok &= good
        print(f"  gop selftest token {i}: {s:.4f}  {'ok' if good else 'FAIL (expected > -0.2)'}")

    return ok


# ───────────────────────────────────────────────────────── main calibration flow

def load_tokens(tokens_path: str) -> dict[str, int]:
    mapping = {}
    with open(tokens_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def run_model(session: "ort.InferenceSession", audio: np.ndarray, sr: int) -> np.ndarray:
    """Returns (T, V) log-probabilities. Audio must be 16 kHz mono float32."""
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz, got {sr} Hz — resample first.")
    wav_t = audio.astype(np.float32).reshape(1, -1)
    len_t = np.array([[audio.shape[0]]], dtype=np.int64)
    out = session.run(None, {"audio_signal": wav_t, "length": len_t})
    logits = out[0][0]   # (T, V)
    return np.array([log_softmax(row) for row in logits])


def calibrate(recordings_dir: str, tokens_path: str, model_path: str):
    if not FULL_MODE:
        print("Skipping calibration — install onnxruntime scipy matplotlib first.")
        return

    tokens = load_tokens(tokens_path)
    session = ort.InferenceSession(
        model_path,
        sess_options=_session_opts()
    )

    correct_gops: list[float] = []
    wrong_gops:   list[float] = []
    skipped = 0

    pattern = re.compile(r"^(.+)_(correct|wrong)_\d+\.wav$")
    for wav_path in sorted(pathlib.Path(recordings_dir).glob("*.wav")):
        m = pattern.match(wav_path.name)
        if not m:
            print(f"  [skip] unrecognised filename: {wav_path.name}")
            skipped += 1
            continue

        glyph, label = m.group(1), m.group(2)
        sr, raw = wav.read(str(wav_path))
        audio = raw.astype(np.float32) / 32768.0

        label_ids = [tokens[ch] for ch in glyph if ch in tokens]
        if not label_ids:
            print(f"  [skip] no tokens for glyph '{glyph}'")
            skipped += 1
            continue

        lp = run_model(session, audio, sr)
        scores = gop(lp, label_ids)
        mean_score = float(np.mean([s for s in scores if s > NEG_INF / 2] or [NEG_INF]))

        if label == "correct":
            correct_gops.append(mean_score)
        else:
            wrong_gops.append(mean_score)

    if not correct_gops and not wrong_gops:
        print("No recordings found — nothing to calibrate.")
        return

    print(f"\nLoaded {len(correct_gops)} correct + {len(wrong_gops)} wrong recordings"
          f" ({skipped} skipped).")

    # ── sweep thresholds to find best F1 ──────────────────────────────────────
    all_scores   = correct_gops + wrong_gops
    all_labels   = [1] * len(correct_gops) + [0] * len(wrong_gops)
    thresholds   = np.linspace(min(all_scores), max(all_scores), 200)
    best_f1, best_t = 0.0, -0.5

    for t in thresholds:
        pred = [1 if s >= t else 0 for s in all_scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(pred, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(pred, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(pred, all_labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    print(f"\nBest threshold: {best_t:.3f}  (F1 = {best_f1:.3f})")
    print(f"Update PhonemeHighlighter.CORRECT_THRESHOLD to: {best_t:.2f}f")

    # ── distribution plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.hist(correct_gops, bins=30, alpha=0.6, label="correct", color="green")
    plt.hist(wrong_gops,   bins=30, alpha=0.6, label="wrong",   color="red")
    plt.axvline(best_t, color="black", linestyle="--", label=f"threshold={best_t:.2f}")
    plt.axvline(-0.50,  color="grey",  linestyle=":",  label="current hardcoded (-0.50)")
    plt.xlabel("Mean GOP score (log scale)")
    plt.ylabel("Count")
    plt.title("GOP distribution — adult Marathi baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gop_distribution.png", dpi=150)
    print("Saved plot to gop_distribution.png")


def _session_opts():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return opts


# ───────────────────────────────────────────────────────── entry point

def main():
    ok = selftest()
    print()
    if not ok:
        print("SELFTEST FAILED — do not trust calibration results.")
        return 1

    print("SELFTEST PASSED\n")

    if not FULL_MODE:
        return 0

    parser = argparse.ArgumentParser(description="Calibrate PhonemeHighlighter threshold")
    parser.add_argument("--recordings", default="data/adult_baseline/",
                        help="Directory of labelled .wav files")
    parser.add_argument("--tokens",     default="assets/asr/tokens.txt")
    parser.add_argument("--model",      default="assets/asr/indicconformer_mr_ctc_int8.onnx")
    args = parser.parse_args()

    if not pathlib.Path(args.recordings).exists():
        print(f"Recordings directory '{args.recordings}' not found — skipping calibration.")
        print("Run with real recordings to calibrate CORRECT_THRESHOLD.")
        return 0
    if not pathlib.Path(args.tokens).exists():
        print(f"Tokens file '{args.tokens}' not found — skipping calibration.")
        return 0
    calibrate(args.recordings, args.tokens, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
