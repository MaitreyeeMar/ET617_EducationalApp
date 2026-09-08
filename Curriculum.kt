package org.nplusone.aksharvel.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * Curriculum — HLD §2.2 progressive pathway: Varnamala -> words -> Jod Akshare.
 *
 * Held in code for Milestone 1 and moved to Room once the client supplies approved
 * content (HLD §15, "Marathi learning content — Pending"). Keeping the shape identical
 * to the future table means that swap is a data load, not a rewrite.
 */
object Curriculum {

    enum class Stage { VARNAMALA_VOWEL, VARNAMALA_CONSONANT, WORD, JOD_AKSHAR }

    data class Item(
        val glyph: String,
        val stage: Stage,
        val aksharas: List<String>,
        val audioAsset: String
    )

    /**
     * Distractors are hand-built, not random. A wrong tap on a visually confusable pair
     * tells the facilitator something a random wrong answer cannot, and it is also the
     * candidate set the speech scorer gets — so this table does double duty.
     */
    private val CONFUSIONS = mapOf(
        "अ" to listOf("आ", "ओ", "उ"), "आ" to listOf("अ", "ओ", "औ"),
        "इ" to listOf("ई", "उ", "ए"), "ई" to listOf("इ", "ऊ", "ऐ"),
        "उ" to listOf("ऊ", "अ", "व"), "ऊ" to listOf("उ", "ऋ", "क"),
        "क" to listOf("फ", "ख", "म"), "ख" to listOf("क", "ष", "रव"),
        "ग" to listOf("ण", "घ", "म"), "घ" to listOf("ध", "ग", "थ"),
        "म" to listOf("भ", "स", "न"), "ल" to listOf("ब", "व", "त"),
        "प" to listOf("य", "ष", "फ"), "ब" to listOf("व", "ल", "य"),
        "त" to listOf("ल", "न", "म"), "द" to listOf("ढ", "उ", "ट")
    )

    val LADDER: List<Item> = listOf(
        Item("अ", Stage.VARNAMALA_VOWEL, listOf("अ"), "audio/mr/a.opus"),
        Item("आ", Stage.VARNAMALA_VOWEL, listOf("आ"), "audio/mr/aa.opus"),
        Item("इ", Stage.VARNAMALA_VOWEL, listOf("इ"), "audio/mr/i.opus"),
        Item("ई", Stage.VARNAMALA_VOWEL, listOf("ई"), "audio/mr/ii.opus"),
        Item("उ", Stage.VARNAMALA_VOWEL, listOf("उ"), "audio/mr/u.opus"),
        Item("ऊ", Stage.VARNAMALA_VOWEL, listOf("ऊ"), "audio/mr/uu.opus"),
        Item("क", Stage.VARNAMALA_CONSONANT, listOf("क"), "audio/mr/ka.opus"),
        Item("ख", Stage.VARNAMALA_CONSONANT, listOf("ख"), "audio/mr/kha.opus"),
        Item("ग", Stage.VARNAMALA_CONSONANT, listOf("ग"), "audio/mr/ga.opus"),
        Item("घ", Stage.VARNAMALA_CONSONANT, listOf("घ"), "audio/mr/gha.opus"),
        Item("म", Stage.VARNAMALA_CONSONANT, listOf("म"), "audio/mr/ma.opus"),
        Item("ल", Stage.VARNAMALA_CONSONANT, listOf("ल"), "audio/mr/la.opus"),
        Item("घर", Stage.WORD, listOf("घ", "र"), "audio/mr/ghar.opus"),
        Item("कमळ", Stage.WORD, listOf("क", "म", "ळ"), "audio/mr/kamal.opus"),
        Item("मासा", Stage.WORD, listOf("मा", "सा"), "audio/mr/masa.opus"),
        Item("झाड", Stage.WORD, listOf("झा", "ड"), "audio/mr/zad.opus")
    )

    fun itemFor(glyph: String): Item? = LADDER.firstOrNull { it.glyph == glyph }

    /** Four options: the target plus three confusables, shuffled. */
    fun optionsFor(glyph: String): List<String> {
        val pool = CONFUSIONS[glyph]
            ?: LADDER.map { it.glyph }.filter { it != glyph && it.length == glyph.length }
        return (listOf(glyph) + pool.take(3)).shuffled()
    }

    /** Mastery gate: 4 of 5 advances, per the Mode specs. */
    const val ITEMS_PER_ROUND = 5
    const val PASS_THRESHOLD = 4
    /** A mis-scoring model must be able to annoy a child, never to block them. */
    const val MAX_ATTEMPTS_BEFORE_ACCEPT = 2
}

// --------------------------------------------------------------------- Room, HLD §8.2

/** HLD §8.2 "User". No password: the mark IS the identity (see wireframe frame 1). */
@Entity(tableName = "learners")
data class Learner(
    @PrimaryKey val id: String,
    val mark: String,                 // 🐟 🌙 🪶 — chosen, never typed
    val createdAt: Long
)

enum class Mode { LISTEN, SPEAK, WRITE }

/** HLD §8.2 "Activity Attempt". */
@Entity(
    tableName = "attempts",
    foreignKeys = [ForeignKey(
        entity = Learner::class, parentColumns = ["id"],
        childColumns = ["learnerId"], onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("learnerId"), Index("glyph")]
)
data class Attempt(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val learnerId: String,
    val glyph: String,
    val mode: Mode,
    val correct: Boolean,
    val detail: String?,              // band name, so the facilitator sees WHY, not just pass/fail
    val attemptNo: Int,
    val timestamp: Long
)

/**
 * HLD §8.2 "Progress Record". Deliberately no raw audio or ink is stored anywhere in
 * this schema — HLD §8.4 forbids retaining raw learner speech or handwriting images,
 * so the classifier verdict is persisted and the input is discarded.
 */
@Entity(tableName = "progress", primaryKeys = ["learnerId", "glyph"])
data class ProgressRecord(
    val learnerId: String,
    val glyph: String,
    val listenPasses: Int = 0,
    val speakPasses: Int = 0,
    val writePasses: Int = 0,
    val mastered: Boolean = false,
    val lastSeen: Long = 0
)

@Dao
interface AksharvelDao {
    @Query("SELECT * FROM learners ORDER BY createdAt")
    fun learners(): Flow<List<Learner>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertLearner(l: Learner)

    @Insert
    suspend fun addAttempt(a: Attempt)

    @Query("SELECT * FROM progress WHERE learnerId = :id")
    fun progress(id: String): Flow<List<ProgressRecord>>

    @Query("SELECT * FROM progress WHERE learnerId = :id AND glyph = :g")
    suspend fun progressFor(id: String, g: String): ProgressRecord?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertProgress(p: ProgressRecord)

    @Query("SELECT COUNT(*) FROM progress WHERE learnerId = :id AND mastered = 1")
    fun masteredCount(id: String): Flow<Int>

    /** Facilitator peek — HLD §5.2 / UC-04. Read-only, no login, opened by long-press. */
    @Query("""
        SELECT l.mark AS mark, a.glyph AS glyph,
               SUM(CASE WHEN a.correct THEN 1 ELSE 0 END) AS passes,
               COUNT(*) AS total
        FROM attempts a JOIN learners l ON l.id = a.learnerId
        WHERE a.timestamp > :since
        GROUP BY l.id, a.glyph ORDER BY l.mark, a.glyph
    """)
    suspend fun weeklySummary(since: Long): List<FacilitatorRow>
}

data class FacilitatorRow(val mark: String, val glyph: String, val passes: Int, val total: Int)

@Database(entities = [Learner::class, Attempt::class, ProgressRecord::class], version = 1)
abstract class AksharvelDb : RoomDatabase() {
    abstract fun dao(): AksharvelDao
}

/** Applies an attempt to the progress record. Mastery = passed in all three modes. */
suspend fun AksharvelDao.record(
    learnerId: String, glyph: String, mode: Mode,
    correct: Boolean, detail: String?, attemptNo: Int
) {
    val now = System.currentTimeMillis()
    addAttempt(Attempt(0, learnerId, glyph, mode, correct, detail, attemptNo, now))
    val cur = progressFor(learnerId, glyph) ?: ProgressRecord(learnerId, glyph)
    val next = cur.copy(
        listenPasses = cur.listenPasses + if (mode == Mode.LISTEN && correct) 1 else 0,
        speakPasses = cur.speakPasses + if (mode == Mode.SPEAK && correct) 1 else 0,
        writePasses = cur.writePasses + if (mode == Mode.WRITE && correct) 1 else 0,
        lastSeen = now
    )
    upsertProgress(next.copy(
        mastered = next.listenPasses > 0 && next.speakPasses > 0 && next.writePasses > 0
    ))
}
