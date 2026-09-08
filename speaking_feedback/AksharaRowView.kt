package org.nplusone.aksharvel.ui.speaking

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator
import androidx.core.content.ContextCompat
import org.nplusone.aksharvel.R

/**
 * AksharaRowView — the phoneme highlight strip for Speaking Mode (बोला).
 *
 * Renders a horizontal row of rounded-rect "chips", one per akshara. Each chip
 * changes colour to match its [PhonemeHighlighter.State]:
 *
 *   PENDING   → mid-grey   (neutral; child has not spoken)
 *   CORRECT   → green      (pronounced acceptably)
 *   WEAK      → amber      (below threshold but not the worst)
 *   FOCUS     → amber + animated bottom-border pulse (the weakest akshara)
 *   UNSCORED  → light grey (model could not align this akshara)
 *
 * Design constraint from MoM: NO TEXT LABELS anywhere on child-facing screens.
 * The akshara glyphs shown inside each chip are the letters themselves, not
 * English descriptions. The colours carry the pedagogical meaning.
 *
 * Sizing: chips self-size to fill the view's width evenly with a fixed gap.
 * Font size scales with chip width, so this works on 480 px and 1080 px screens
 * without a separate layout for each density bucket.
 *
 * Usage (XML):
 *   <org.nplusone.aksharvel.ui.speaking.AksharaRowView
 *       android:id="@+id/aksharaRow"
 *       android:layout_width="match_parent"
 *       android:layout_height="72dp" />
 *
 * Usage (code):
 *   aksharaRow.setStates(viewModel.uiState.value.aksharaStates)
 */
class AksharaRowView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var states: List<AksharaState> = emptyList()

    // ---------------------------------------------------------------- paints

    private val chipPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        // Noto Sans Devanagari is bundled in assets/fonts/ by Khushbu's wireframe module.
        // Fall back to the system Devanagari face if the asset is not yet present.
        typeface = try {
            android.graphics.Typeface.createFromAsset(context.assets, "fonts/NotoSansDevanagari-Regular.ttf")
        } catch (_: Exception) {
            android.graphics.Typeface.DEFAULT
        }
    }
    private val focusBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    // ---------------------------------------------------------------- colours

    private val colourPending   = resolveColour(R.color.aksharvel_chip_pending)
    private val colourCorrect   = resolveColour(R.color.aksharvel_chip_correct)
    private val colourWeak      = resolveColour(R.color.aksharvel_chip_weak)
    private val colourUnscored  = resolveColour(R.color.aksharvel_chip_unscored)
    private val colourText      = resolveColour(R.color.aksharvel_chip_text)
    private val colourFocusBorder = resolveColour(R.color.aksharvel_chip_focus_border)

    private fun resolveColour(resId: Int) = ContextCompat.getColor(context, resId)

    // ---------------------------------------------------------------- focus animation

    /** Animates the alpha of the focus-border pulse: 1.0 → 0.3 → 1.0, looping. */
    private var focusAlpha = 255
    private val focusAnimator = ValueAnimator.ofInt(255, 76, 255).apply {
        duration = 900
        repeatCount = ValueAnimator.INFINITE
        interpolator = DecelerateInterpolator()
        addUpdateListener {
            focusAlpha = it.animatedValue as Int
            invalidate()
        }
    }

    // ---------------------------------------------------------------- layout helpers

    private val gap = dpToPx(6f)
    private val cornerRadius = dpToPx(8f)
    private val chipRect = RectF()

    private fun dpToPx(dp: Float) = dp * context.resources.displayMetrics.density

    // ---------------------------------------------------------------- public API

    /** Replaces the displayed states and triggers a redraw + animation reset. */
    fun setStates(newStates: List<AksharaState>) {
        states = newStates
        val hasFocus = newStates.any { it.state == PhonemeHighlighter.State.FOCUS }
        if (hasFocus && !focusAnimator.isRunning) {
            focusAnimator.start()
        } else if (!hasFocus && focusAnimator.isRunning) {
            focusAnimator.cancel()
            focusAlpha = 255
        }
        invalidate()
    }

    // ---------------------------------------------------------------- drawing

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (states.isEmpty()) return

        val n = states.size
        val totalGap = gap * (n + 1)
        val chipW = (width - totalGap) / n
        val chipH = height - gap * 2
        textPaint.textSize = chipW * 0.52f          // scales with chip size

        states.forEachIndexed { i, as_ ->
            val left  = gap + i * (chipW + gap)
            val top   = gap
            chipRect.set(left, top, left + chipW, top + chipH)

            // Background
            chipPaint.color = when (as_.state) {
                PhonemeHighlighter.State.PENDING   -> colourPending
                PhonemeHighlighter.State.CORRECT   -> colourCorrect
                PhonemeHighlighter.State.WEAK      -> colourWeak
                PhonemeHighlighter.State.FOCUS     -> colourWeak      // same bg as WEAK
                PhonemeHighlighter.State.UNSCORED  -> colourUnscored
            }
            chipPaint.style = Paint.Style.FILL
            canvas.drawRoundRect(chipRect, cornerRadius, cornerRadius, chipPaint)

            // Focus border pulse
            if (as_.state == PhonemeHighlighter.State.FOCUS) {
                focusBorderPaint.color = colourFocusBorder
                focusBorderPaint.alpha = focusAlpha
                canvas.drawRoundRect(chipRect, cornerRadius, cornerRadius, focusBorderPaint)
            }

            // Akshara glyph
            textPaint.color = colourText
            val textY = chipRect.centerY() - (textPaint.ascent() + textPaint.descent()) / 2f
            canvas.drawText(as_.akshara, chipRect.centerX(), textY, textPaint)
        }
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        focusAnimator.cancel()
    }
}
