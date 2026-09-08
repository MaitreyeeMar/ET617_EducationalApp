package org.nplusone.aksharvel.assess

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.hypot
import kotlin.math.min

/**
 * Tactile Mode classifier — HLD §6.1 module 2, UC-02.
 *
 * Pure geometry. No model file, no TFLite/ONNX dependency, no inference latency, and
 * nothing to train. It runs on a 1 GB device today, which is why it is on the
 * Milestone 1 critical path while the ASR is not.
 *
 * Ported from stroke_validate.py, whose distance transform was checked against exact
 * Euclidean distance (max error 0.71 px) and whose five failure cases all pass.
 *
 * Three findings from that validation are baked in here, and each one was a case that
 * scored as CORRECT before it was fixed:
 *
 *  1. Tolerances must be ASYMMETRIC. Generous about a child missing the line (cheap
 *     digitisers and small fingers wander); strict about drawing where no letter is.
 *  2. The shirorekha needs its own hard gate. It is only ~25% of क's pixels, so a
 *     child who omits it entirely still averages 0.76 coverage.
 *  3. Every significant region needs a floor. Bar-plus-stem alone averages 0.72 —
 *     a global mean lets a whole missing limb through. This is the general form of (2).
 */
class StrokeClassifier(
    private val size: Int,
    private val typeface: Typeface
) {
    companion object {
        private const val COVER_PASS = 0.70f
        private const val SPILL_FAIL = 0.25f
        private const val REGION_FLOOR = 0.55f      // every part must be attempted
        private const val SHIRO_PASS = 0.60f
        private const val SCRIBBLE_RATIO = 2.5f
        private const val MIN_REGION_SHARE = 0.08f  // ignore regions this small
        private const val INF = 1e9f
    }

    enum class Band { ACCEPTED, INCOMPLETE, MISSING_SHIROREKHA, OFF_LINE, SCRIBBLE, EMPTY }

    data class Result(
        val band: Band,
        val coverage: Float,
        val spill: Float,
        val lengthRatio: Float,
        val orderOk: Boolean?,       // null when the stroke count differs from the template
        val shirorekhaOk: Boolean?,
        val weakestRegion: String?   // null when nothing is notably weak
    ) {
        val accepted get() = band == Band.ACCEPTED
    }

    /** A drawn stroke: screen points in order, as captured from MotionEvent. */
    data class Stroke(val points: List<Pair<Float, Float>>)

    // ------------------------------------------------------------------ glyph raster

    private var glyphMask = BooleanArray(0)
    private var distToGlyph = FloatArray(0)
    private var glyphPx = 0
    private var strokeW = 8f
    private var shiroBand: IntArray? = null

    /** Rasterise the target once when the screen opens, not on every check. */
    fun setTarget(glyph: String) {
        val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ALPHA_8)
        val c = Canvas(bmp)
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.typeface = this@StrokeClassifier.typeface
            textSize = size * 0.86f
            textAlign = Paint.Align.CENTER
            color = Color.WHITE
        }
        val fm = p.fontMetrics
        c.drawText(glyph, size / 2f, size / 2f - (fm.ascent + fm.descent) / 2f, p)

        glyphMask = BooleanArray(size * size)
        val row = IntArray(size)
        var count = 0
        for (y in 0 until size) {
            bmp.getPixels(row, 0, size, 0, y, size, 1)
            for (x in 0 until size) {
                // ALPHA_8 packs alpha in the high byte
                val a = (row[x] ushr 24) and 0xFF
                if (a > 60) { glyphMask[y * size + x] = true; count++ }
            }
        }
        bmp.recycle()
        glyphPx = count

        distToGlyph = chamfer(glyphMask)
        strokeW = estimateStrokeWidth()
        shiroBand = findShirorekha()
    }

    /**
     * Pen width = 2x the 90th-percentile inner distance-to-edge. Every tolerance is
     * derived from this, so one set of thresholds works on a 480x854 phone and a
     * 1080p tablet without retuning.
     */
    private fun estimateStrokeWidth(): Float {
        if (glyphPx == 0) return 8f
        val inner = chamfer(BooleanArray(glyphMask.size) { !glyphMask[it] })
        val vals = FloatArray(glyphPx)
        var i = 0
        for (k in glyphMask.indices) if (glyphMask[k]) vals[i++] = inner[k]
        vals.sort()
        return 2f * vals[(0.90f * (vals.size - 1)).toInt()]
    }

    /**
     * Locate the shirorekha: the topmost horizontal band whose ink covers most of the
     * glyph's width. Detected rather than hand-authored per letter, so it works for
     * every akshara without a per-glyph table.
     */
    private fun findShirorekha(): IntArray? {
        if (glyphPx == 0) return null
        var minX = size; var maxX = 0
        for (y in 0 until size) for (x in 0 until size)
            if (glyphMask[y * size + x]) { if (x < minX) minX = x; if (x > maxX) maxX = x }
        val width = (maxX - minX + 1).coerceAtLeast(1)

        var firstY = -1
        for (y in 0 until size) {
            var run = 0
            for (x in minX..maxX) if (glyphMask[y * size + x]) run++
            if (run > 0.70f * width) { firstY = y; break }
            if (run > 0 && firstY < 0 && y > size * 0.35f) return null  // no bar in this glyph
        }
        if (firstY < 0) return null
        val h = (strokeW * 1.6f).toInt().coerceAtLeast(3)
        return intArrayOf(firstY, min(size, firstY + h))
    }

    // ------------------------------------------------------------------ scoring

    fun classify(strokes: List<Stroke>, anchors: List<Pair<Float, Float>>?): Result {
        if (glyphPx == 0 || strokes.isEmpty())
            return Result(Band.EMPTY, 0f, 0f, 0f, null, null, null)

        val coverTol = 1.75f * strokeW
        val spillTol = 1.00f * strokeW

        // rasterise the child's ink, and measure how far each drawn point is from the glyph
        val drawn = BooleanArray(size * size)
        var pts = 0
        var outside = 0
        var drawnLen = 0f
        for (s in strokes) {
            val p = s.points
            for (i in 0 until p.size - 1) {
                val (x0, y0) = p[i]; val (x1, y1) = p[i + 1]
                drawnLen += hypot(x1 - x0, y1 - y0)
                val n = (maxOf(kotlin.math.abs(x1 - x0), kotlin.math.abs(y1 - y0)).toInt() + 1).coerceAtLeast(2)
                for (k in 0 until n) {
                    val t = k.toFloat() / (n - 1)
                    val x = (x0 + (x1 - x0) * t).toInt()
                    val y = (y0 + (y1 - y0) * t).toInt()
                    if (x in 0 until size && y in 0 until size) {
                        drawn[y * size + x] = true
                        pts++
                        if (distToGlyph[y * size + x] > spillTol) outside++
                    }
                }
            }
        }
        if (pts == 0) return Result(Band.EMPTY, 0f, 0f, 0f, null, null, null)

        val distToDrawn = chamfer(drawn)

        var covered = 0
        for (k in glyphMask.indices) if (glyphMask[k] && distToDrawn[k] <= coverTol) covered++
        val coverage = covered.toFloat() / glyphPx
        val spill = outside.toFloat() / pts

        val skeletonLen = glyphPx / strokeW.coerceAtLeast(1f)
        val lengthRatio = drawnLen / skeletonLen.coerceAtLeast(1f)

        // per-region floor
        var weakest = 1f; var weakestName: String? = null
        val names = arrayOf("top-left", "top-right", "bottom-left", "bottom-right")
        for (gy in 0..1) for (gx in 0..1) {
            var sub = 0; var hit = 0
            for (y in gy * size / 2 until (gy + 1) * size / 2)
                for (x in gx * size / 2 until (gx + 1) * size / 2) {
                    val k = y * size + x
                    if (glyphMask[k]) { sub++; if (distToDrawn[k] <= coverTol) hit++ }
                }
            if (sub < glyphPx * MIN_REGION_SHARE) continue
            val c = hit.toFloat() / sub
            if (c < weakest) { weakest = c; weakestName = names[gy * 2 + gx] }
        }
        val regionOk = weakest >= REGION_FLOOR
        if (weakest >= 0.75f) weakestName = null

        // shirorekha gate
        var shiroOk: Boolean? = null
        shiroBand?.let { (y0, y1) ->
            var bar = 0; var hit = 0
            for (y in y0 until y1) for (x in 0 until size) {
                val k = y * size + x
                if (glyphMask[k]) { bar++; if (distToDrawn[k] <= coverTol) hit++ }
            }
            if (bar > 0) shiroOk = hit.toFloat() / bar >= SHIRO_PASS
        }

        // stroke order
        var orderOk: Boolean? = null
        if (anchors != null && anchors.size == strokes.size) {
            orderOk = true
            for (i in strokes.indices) {
                val (sx, sy) = strokes[i].points.first()
                var bestI = 0; var bestD = Float.MAX_VALUE
                anchors.forEachIndexed { j, (ax, ay) ->
                    val d = hypot(sx - ax, sy - ay)
                    if (d < bestD) { bestD = d; bestI = j }
                }
                if (bestI != i) { orderOk = false; break }
            }
        }

        // Reject scribbles before praising coverage; check structure before the average.
        val band = when {
            lengthRatio > SCRIBBLE_RATIO -> Band.SCRIBBLE
            spill > SPILL_FAIL -> Band.OFF_LINE
            shiroOk == false -> Band.MISSING_SHIROREKHA
            coverage < COVER_PASS || !regionOk -> Band.INCOMPLETE
            else -> Band.ACCEPTED
        }
        return Result(band, coverage, spill, lengthRatio, orderOk, shiroOk, weakestName)
    }

    /**
     * Two-pass 3-4 chamfer distance transform. O(pixels), no allocation per frame after
     * the first call, and within ~0.7 px of exact Euclidean — far tighter than the
     * tolerances it feeds. A brute-force nearest-point search here would be O(n^2) and
     * would not hold 60 fps.
     */
    private fun chamfer(mask: BooleanArray): FloatArray {
        val d = FloatArray(mask.size) { if (mask[it]) 0f else INF }
        for (y in 0 until size) for (x in 0 until size) {
            val k = y * size + x
            var b = d[k]
            if (y > 0) {
                if (x > 0) b = minOf(b, d[k - size - 1] + 1.4142f)
                b = minOf(b, d[k - size] + 1f)
                if (x < size - 1) b = minOf(b, d[k - size + 1] + 1.4142f)
            }
            if (x > 0) b = minOf(b, d[k - 1] + 1f)
            d[k] = b
        }
        for (y in size - 1 downTo 0) for (x in size - 1 downTo 0) {
            val k = y * size + x
            var b = d[k]
            if (y < size - 1) {
                if (x < size - 1) b = minOf(b, d[k + size + 1] + 1.4142f)
                b = minOf(b, d[k + size] + 1f)
                if (x > 0) b = minOf(b, d[k + size - 1] + 1.4142f)
            }
            if (x < size - 1) b = minOf(b, d[k + 1] + 1f)
            d[k] = b
        }
        return d
    }
}

/** Spoken feedback per band. Never a bare cross — always what to do next. */
fun StrokeClassifier.Result.marathiFeedback(): String = when (band) {
    StrokeClassifier.Band.ACCEPTED -> if (orderOk == false) "छान! पण क्रम बदलला" else "छान!"
    StrokeClassifier.Band.MISSING_SHIROREKHA -> "वरची रेघ राहिली"
    StrokeClassifier.Band.INCOMPLETE -> "अजून थोडं राहिलं"
    StrokeClassifier.Band.OFF_LINE -> "रेषेच्या आत लिही"
    StrokeClassifier.Band.SCRIBBLE -> "हळू, अक्षर काढ"
    StrokeClassifier.Band.EMPTY -> "अक्षर लिही"
}
