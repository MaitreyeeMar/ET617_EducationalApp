package org.nplusone.aksharvel.speech

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p

/**
 * On-device Marathi pronunciation check for Aksharvel, Mode 2 (बोला).
 *
 * Model:  ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large — CTC head only,
 *         exported to ONNX and INT8-quantised (~120-130 MB). MIT licence.
 *
 * IMPORTANT — export the model with NeMo's AudioToMelSpectrogramPreprocessor FUSED
 * INTO THE GRAPH so this file can feed raw 16 kHz float samples. Do not reimplement
 * log-mel extraction in Kotlin. Feature mismatch between your Python validation and
 * your Android build is the single most common way this pipeline silently produces
 * garbage, and it is entirely avoidable.
 *
 * The scoring below is a direct port of pron_score.py, whose CTC forward pass was
 * verified against brute-force alignment enumeration.
 *
 * Reference implementation — not compiled. Verify tensor names against your own
 * export with `Netron` before wiring it up.
 */
class PronunciationChecker(context: Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    /** token string -> id, loaded from the export's tokens.txt. Blank is id 0 in NeMo CTC. */
    private val tokenToId: Map<String, Int>
    private val blank = 0

    init {
        val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)          // 2 is right for a Snapdragon 4xx; 4 starves the UI
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Deliberately NOT calling addNnapi(): NNAPI on budget SoCs frequently falls back
            // to CPU per-op and ends up slower than plain CPU for INT8 conformers. Measure on
            // your actual test devices before enabling it.
        }
        session = env.createSession(bytes, opts)

        tokenToId = context.assets.open(TOKENS_ASSET).bufferedReader().useLines { lines ->
            lines.mapNotNull { line ->
                val parts = line.trim().split(" ")
                if (parts.size >= 2) parts[0] to parts[1].toInt() else null
            }.toMap()
        }
    }

    // ---------------------------------------------------------------- capture

    /**
     * Records until [stop] flips true or [maxMs] elapses, whichever comes first.
     * Returns null when nothing above the noise floor was captured — that is a mic
     * or environment problem and must never be reported to the child as a mistake.
     */
    fun record(stop: () -> Boolean, maxMs: Int = 4000, onLevel: (Float) -> Unit): FloatArray? {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,   // gives you the platform AEC/NS
            SAMPLE_RATE, CHANNEL, ENCODING, minBuf * 4
        )
        val out = ArrayList<Float>(SAMPLE_RATE * maxMs / 1000)
        val buf = ShortArray(minBuf / 2)
        var peak = 0f
        try {
            rec.startRecording()
            val deadline = System.currentTimeMillis() + maxMs
            while (!stop() && System.currentTimeMillis() < deadline) {
                val n = rec.read(buf, 0, buf.size)
                var frameMax = 0f
                for (i in 0 until n) {
                    val v = buf[i] / 32768f
                    out.add(v)
                    if (abs(v) > frameMax) frameMax = abs(v)
                }
                if (frameMax > peak) peak = frameMax
                onLevel(frameMax)
            }
        } finally {
            rec.stop(); rec.release()
        }
        // 0.02 ≈ -34 dBFS. Calibrate in an actual ZP classroom, not at a desk — the
        // ambient floor in a room with 40 children is nothing like a quiet office.
        return if (peak < 0.02f) null else out.toFloatArray()
    }

    // ---------------------------------------------------------------- inference

    /** Runs the encoder + CTC head. Returns (T, V) log-probabilities. */
    private fun logProbs(audio: FloatArray): Array<FloatArray> {
        val wav = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), longArrayOf(1, audio.size.toLong()))
        val len = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(longArrayOf(audio.size.toLong())), longArrayOf(1))
        wav.use { _ -> len.use { _ ->
            session.run(mapOf("audio_signal" to wav, "length" to len)).use { res ->
                @Suppress("UNCHECKED_CAST")
                val raw = (res[0].value as Array<Array<FloatArray>>)[0]   // (T, V) logits
                return Array(raw.size) { t -> logSoftmax(raw[t]) }
            }
        } }
    }

    private fun logSoftmax(x: FloatArray): FloatArray {
        val m = x.max()
        var s = 0.0
        for (v in x) s += exp((v - m).toDouble())
        val ls = m + ln(s).toFloat()
        return FloatArray(x.size) { x[it] - ls }
    }

    // ---------------------------------------------------------------- scoring

    /**
     * The whole point: score only the words that were on screen. Open-vocabulary
     * decoding plus string comparison is what mislabels accented children.
     */
    suspend fun check(
        audio: FloatArray,
        target: String,
        onScreen: List<String>,
        aksharas: List<String>,
        margin: Float = 0.35f,
        gopFloor: Float = -1.2f
    ): Verdict = withContext(Dispatchers.Default) {
        val lp = logProbs(audio)

        val scored = onScreen.associateWith { w ->
            val toks = tokenise(w)
            if (toks.isEmpty()) NEG_INF else ctcLogProb(lp, toks) / toks.size
        }
        val ranked = scored.entries.sortedByDescending { it.value }
        val best = ranked[0].key
        val gap = ranked[0].value - (ranked.getOrNull(1)?.value ?: NEG_INF)

        val targetToks = tokenise(target)
        val g = gop(lp, targetToks)
        val worstIdx = g.indices.minByOrNull { g[it] } ?: -1
        val worst = aksharas.getOrNull(worstIdx)

        val band = when {
            best == target && gap >= margin && (g.minOrNull() ?: NEG_INF) >= gopFloor -> Band.ACCEPTED
            else -> Band.RETRY
        }
        Verdict(band, best, gap, g, worst)
    }

    /** Split into the export's token inventory. Char-level for IndicConformer's Devanagari set. */
    private fun tokenise(word: String): IntArray =
        word.map { tokenToId[it.toString()] ?: -1 }.filter { it >= 0 }.toIntArray()

    private fun logAdd(a: Float, b: Float): Float {
        if (a <= NEG_INF / 2) return b
        if (b <= NEG_INF / 2) return a
        val (hi, lo) = if (a > b) a to b else b to a
        return hi + ln1p(exp((lo - hi).toDouble())).toFloat()
    }

    /** Exact log P(labels | audio) under CTC. Verified against brute force in pron_score.py. */
    private fun ctcLogProb(lp: Array<FloatArray>, labels: IntArray): Float {
        val t = lp.size
        if (t < labels.size) return NEG_INF
        val ext = IntArray(labels.size * 2 + 1) { blank }
        for (i in labels.indices) ext[i * 2 + 1] = labels[i]
        val s = ext.size

        var a = FloatArray(s) { NEG_INF }
        a[0] = lp[0][ext[0]]
        if (s > 1) a[1] = lp[0][ext[1]]

        for (frame in 1 until t) {
            val nxt = FloatArray(s) { NEG_INF }
            val lo = maxOf(0, s - 2 * (t - frame))
            for (i in lo until s) {
                var v = a[i]
                if (i > 0) v = logAdd(v, a[i - 1])
                if (i > 1 && ext[i] != blank && ext[i] != ext[i - 2]) v = logAdd(v, a[i - 2])
                nxt[i] = v + lp[frame][ext[i]]
            }
            a = nxt
        }
        return logAdd(a[s - 1], if (s > 1) a[s - 2] else NEG_INF)
    }

    /** Per-akshara Goodness of Pronunciation. 0 is perfect; more negative is worse. */
    private fun gop(lp: Array<FloatArray>, labels: IntArray): FloatArray {
        if (labels.isEmpty() || lp.size < labels.size) return FloatArray(labels.size) { NEG_INF }
        val t = lp.size
        val ext = IntArray(labels.size * 2 + 1) { blank }
        for (i in labels.indices) ext[i * 2 + 1] = labels[i]
        val s = ext.size

        val d = Array(t) { FloatArray(s) { NEG_INF } }
        val bp = Array(t) { IntArray(s) }
        d[0][0] = lp[0][ext[0]]
        if (s > 1) d[0][1] = lp[0][ext[1]]

        for (frame in 1 until t) for (i in 0 until s) {
            var bestPrev = d[frame - 1][i]; var bestI = i
            if (i > 0 && d[frame - 1][i - 1] > bestPrev) { bestPrev = d[frame - 1][i - 1]; bestI = i - 1 }
            if (i > 1 && ext[i] != blank && ext[i] != ext[i - 2] && d[frame - 1][i - 2] > bestPrev) {
                bestPrev = d[frame - 1][i - 2]; bestI = i - 2
            }
            d[frame][i] = bestPrev + lp[frame][ext[i]]
            bp[frame][i] = bestI
        }

        var cur = if (s == 1 || d[t - 1][s - 1] >= d[t - 1][s - 2]) s - 1 else s - 2
        val path = IntArray(t)
        for (frame in t - 1 downTo 0) { path[frame] = cur; cur = bp[frame][cur] }

        val sum = FloatArray(labels.size); val cnt = IntArray(labels.size)
        for (frame in 0 until t) {
            val i = path[frame]
            if (ext[i] == blank) continue
            val li = (i - 1) / 2
            sum[li] += lp[frame][ext[i]] - lp[frame].max()
            cnt[li]++
        }
        return FloatArray(labels.size) { if (cnt[it] == 0) NEG_INF else sum[it] / cnt[it] }
    }

    override fun close() { session.close() }

    enum class Band { ACCEPTED, RETRY, NO_SPEECH }

    data class Verdict(
        val band: Band,
        val heardAs: String,
        val margin: Float,
        val gop: FloatArray,
        val weakestAkshara: String?
    )

    companion object {
        private const val MODEL_ASSET = "asr/indicconformer_mr_ctc_int8.onnx"
        private const val TOKENS_ASSET = "asr/tokens.txt"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        private const val NEG_INF = -1e30f
    }
}
