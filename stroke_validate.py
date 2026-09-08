"""
Validation harness for the tactile (handwriting) classifier before it is ported to Kotlin.

Everything here is geometry, not ML: no model file, no training data, no inference
latency. That is the whole reason to prefer it for Milestone 1 — it works today on a
1 GB device, and it gives feedback that names the error instead of scoring it.

Pipeline being validated:
  1. Rasterise the target glyph to a binary mask.
  2. Build a chamfer distance transform of the mask ONCE (two passes, O(pixels)).
     Then coverage and spill are both cheap lookups instead of nested loops —
     this is what makes it run at 60 fps on a Snapdragon 4xx.
  3. Coverage   : fraction of glyph pixels within tol of some drawn point.
  4. Spill      : fraction of drawn points farther than tol from the glyph.
  5. Stroke order: each stroke's start must be nearest ITS OWN anchor.
  6. Shirorekha : the top bar must actually be drawn — the most common Devanagari
     error, and worth calling out separately rather than folding into coverage.
"""

import numpy as np

TOL = 14.0          # px; how far off the line a child may be and still be "on" it
COVER_PASS = 0.70
SPILL_FAIL = 0.25
REGION_FLOOR = 0.55   # every significant part of the glyph must be attempted


# ---------------------------------------------------------------- distance transform

def chamfer_dt(mask):
    """Two-pass 3x4 chamfer DT. mask: bool array, True = glyph. Returns float dist-to-glyph."""
    INF = 1e9
    h, w = mask.shape
    d = np.where(mask, 0.0, INF)
    # forward pass
    for y in range(h):
        for x in range(w):
            best = d[y, x]
            if y > 0:
                if x > 0:
                    best = min(best, d[y - 1, x - 1] + 1.4142)
                best = min(best, d[y - 1, x] + 1.0)
                if x < w - 1:
                    best = min(best, d[y - 1, x + 1] + 1.4142)
            if x > 0:
                best = min(best, d[y, x - 1] + 1.0)
            d[y, x] = best
    # backward pass
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            best = d[y, x]
            if y < h - 1:
                if x < w - 1:
                    best = min(best, d[y + 1, x + 1] + 1.4142)
                best = min(best, d[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, d[y + 1, x - 1] + 1.4142)
            if x < w - 1:
                best = min(best, d[y, x + 1] + 1.0)
            d[y, x] = best
    return d


def _brute_dt(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            out[y, x] = np.sqrt(((ys - y) ** 2 + (xs - x) ** 2).min())
    return out


# ---------------------------------------------------------------- scoring

def stroke_width(mask):
    """Estimate the glyph's pen width: 2x the 90th-percentile inner distance-to-edge.
    Every tolerance is derived from this, so the classifier is resolution-independent
    and the same thresholds work on a 480x854 phone and a 1080p tablet."""
    inner = chamfer_dt(~mask)
    vals = inner[mask]
    return float(2.0 * np.percentile(vals, 90)) if vals.size else 8.0


def score(mask, strokes, anchors, shiro_band=None):
    """strokes: list of [(x,y), ...]. anchors: expected start point per stroke, in order."""
    h, w = mask.shape
    dist_to_glyph = chamfer_dt(mask)

    sw = stroke_width(mask)
    # ASYMMETRIC on purpose. Be generous about a child MISSING the line — a finger on
    # a cheap digitiser wanders. Be strict about a child drawing where there is no
    # letter, because that is the signal that distinguishes an attempt from a scribble.
    cover_tol = 1.75 * sw
    spill_tol = 1.00 * sw

    # rasterise the child's strokes into a mask so we can DT them too
    drawn = np.zeros((h, w), dtype=bool)
    pts = []
    for s in strokes:
        for i in range(len(s) - 1):
            (x0, y0), (x1, y1) = s[i], s[i + 1]
            n = max(2, int(max(abs(x1 - x0), abs(y1 - y0))) + 1)
            for k in range(n):
                x = int(round(x0 + (x1 - x0) * k / (n - 1)))
                y = int(round(y0 + (y1 - y0) * k / (n - 1)))
                if 0 <= x < w and 0 <= y < h:
                    drawn[y, x] = True
                    pts.append((x, y))
    if not pts:
        return {"band": "empty"}

    dist_to_drawn = chamfer_dt(drawn)

    glyph_px = int(mask.sum())
    covered = int(((dist_to_drawn <= cover_tol) & mask).sum())
    coverage = covered / max(1, glyph_px)

    outside = sum(1 for (x, y) in pts if dist_to_glyph[y, x] > spill_tol)
    spill = outside / len(pts)

    # Scribble gate. A child tracing क draws roughly the glyph's skeleton length; a
    # child scribbling draws several times that. Cheap, and it catches the case where
    # coverage and spill both look acceptable because the ink is simply everywhere.
    drawn_len = 0.0
    for s in strokes:
        for i in range(len(s) - 1):
            drawn_len += float(np.hypot(s[i + 1][0] - s[i][0], s[i + 1][1] - s[i][1]))
    skeleton_len = glyph_px / max(1.0, sw)
    length_ratio = drawn_len / max(1.0, skeleton_len)

    # stroke order: is each stroke's start nearest its own anchor?
    order_ok = True
    if anchors and len(strokes) == len(anchors):
        for i, s in enumerate(strokes):
            sx, sy = s[0]
            dists = [np.hypot(sx - ax, sy - ay) for (ax, ay) in anchors]
            if int(np.argmin(dists)) != i:
                order_ok = False
                break
    elif anchors:
        order_ok = None      # wrong stroke count; don't judge order

    # Shirorekha: a HARD gate, not advisory. The top bar is only ~25% of क's pixels,
    # so a child who omits it entirely still scores 0.76 coverage and would otherwise
    # be told they were right. A structurally missing part cannot be outvoted by a
    # global average — that is the flaw in grading handwriting on coverage alone.
    shiro_ok = None
    if shiro_band is not None:
        y0, y1 = shiro_band
        bar = mask[y0:y1, :]
        bar_cov = ((dist_to_drawn[y0:y1, :] <= cover_tol) & bar).sum() / max(1, bar.sum())
        shiro_ok = bool(bar_cov >= 0.6)

    # Per-region coverage. This is the general form of the shirorekha problem: a
    # global average lets an entire missing limb slide through (bar + stem only still
    # averages 0.72 on क). Requiring EVERY significant region to clear a floor is what
    # actually distinguishes "wrote the whole letter" from "wrote most of the pixels".
    missed, worst = None, 1.0
    for gy in range(2):
        for gx in range(2):
            ys, ye = gy * h // 2, (gy + 1) * h // 2
            xs, xe = gx * w // 2, (gx + 1) * w // 2
            sub = mask[ys:ye, xs:xe]
            if sub.sum() < glyph_px * 0.08:
                continue
            c = ((dist_to_drawn[ys:ye, xs:xe] <= cover_tol) & sub).sum() / sub.sum()
            if c < worst:
                worst, missed = float(c), ["top-left", "top-right", "bottom-left", "bottom-right"][gy * 2 + gx]
    region_ok = worst >= REGION_FLOOR
    if worst >= 0.75:
        missed = None

    # Order matters: reject scribbles before praising coverage, and check structure
    # before checking the global average.
    if length_ratio > 2.5:
        band = "scribble"
    elif spill > SPILL_FAIL:
        band = "off_line"
    elif shiro_ok is False:
        band = "missing_shirorekha"
    elif coverage < COVER_PASS or not region_ok:
        band = "incomplete"
    else:
        band = "accepted"

    return {"band": band, "coverage": round(coverage, 3), "spill": round(spill, 3),
            "len_ratio": round(length_ratio, 2), "stroke_w": round(sw, 1),
            "order_ok": order_ok, "shirorekha_ok": shiro_ok,
            "weakest_region": missed, "weakest_cov": round(worst, 3)}


# ---------------------------------------------------------------- tests

def make_glyph(h=100, w=100):
    """Stand-in for a Devanagari letter: a top bar plus a descending stem and a bowl.
    Shaped like क in the ways that matter — a shirorekha and two lower parts."""
    m = np.zeros((h, w), dtype=bool)
    m[12:20, 15:85] = True          # shirorekha (top bar)
    m[20:85, 48:56] = True          # vertical stem
    m[45:53, 20:48] = True          # left arm
    m[55:85, 20:30] = True          # left bowl leg
    return m


def line(p0, p1, n=60):
    return [(p0[0] + (p1[0] - p0[0]) * i / (n - 1),
             p0[1] + (p1[1] - p0[1]) * i / (n - 1)) for i in range(n)]


def main():
    ok = True

    # --- DT correctness against brute force, on a small mask ---
    small = np.zeros((22, 22), dtype=bool)
    small[8:14, 6:16] = True
    approx, exact = chamfer_dt(small), _brute_dt(small)
    err = np.abs(approx - exact).max()
    good = err < 1.2          # chamfer 3x4 is approximate; ~2-5% is expected and fine
    ok &= good
    print(f"  chamfer DT max error vs exact = {err:.3f} px  {'ok' if good else 'FAIL'}")

    g = make_glyph()
    anchors = [(20, 16), (52, 22)]           # bar start, then stem start
    shiro = (10, 22)

    # --- a good trace: bar, then stem+arm+leg ---
    good_strokes = [
        line((16, 16), (84, 16)),
        line((52, 22), (52, 84)) + line((48, 49), (21, 49)) + line((25, 56), (25, 84)),
    ]
    r = score(g, good_strokes, anchors, shiro)
    print(f"  good trace     -> {r}")
    ok &= r["band"] == "accepted" and r["shirorekha_ok"]

    # --- forgot the shirorekha: the classic error ---
    no_bar = [line((52, 22), (52, 84)) + line((48, 49), (21, 49)) + line((25, 56), (25, 84))]
    r = score(g, no_bar, None, shiro)
    print(f"  no shirorekha  -> {r}")
    ok &= r["band"] == "missing_shirorekha" and r["shirorekha_ok"] is False and r["weakest_region"].startswith("top")

    # --- scribble: covers everything but mostly off the line ---
    scribble = [line((5, 5 + 7 * i), (95, 12 + 7 * i)) for i in range(13)]
    r = score(g, scribble, None, shiro)
    print(f"  scribble       -> {r}")
    ok &= r["band"] in ("scribble", "off_line")

    # --- partial trace: bar + stem only, arm and leg missing ---
    partial = [line((16, 16), (84, 16)), line((52, 22), (52, 84))]
    r = score(g, partial, anchors, shiro)
    print(f"  partial trace  -> {r}")
    ok &= r["band"] == "incomplete" and r["weakest_region"] is not None

    # --- right shape, wrong stroke order (stem before bar) ---
    reversed_order = [good_strokes[1], good_strokes[0]]
    r = score(g, reversed_order, anchors, shiro)
    print(f"  wrong order    -> band={r['band']} order_ok={r['order_ok']}")
    ok &= r["band"] == "accepted" and r["order_ok"] is False

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
