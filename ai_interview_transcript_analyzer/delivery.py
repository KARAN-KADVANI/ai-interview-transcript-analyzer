"""
delivery.py
Rule-based analysis of HOW the candidate speaks (not WHAT they say).

Why rule-based and not a trained model?
  The Kaggle interview datasets contain written model answers, not spoken
  transcripts, and have no "confident / hesitant" labels. Training a classifier
  on invented labels would give fake accuracy numbers. So this part is a
  transparent, explainable lexicon scorer: every point added or removed can be
  traced to a visible cue (filler words, hedges, repeated words, action verbs,
  numbers). The weights are hand-set and listed in SCORE_WEIGHTS.
"""
import re

SCORE_WEIGHTS = {
    "base": 62,
    "filler_penalty_per_100_words": 4.0,   # capped at 35
    "hedge_penalty_per_100_words": 4.5,    # capped at 35
    "repeat_penalty_each": 5.0,            # capped at 15
    "action_verb_bonus_each": 5.0,         # capped at 15
    "has_numbers_bonus": 6.0,
    "very_short_penalty (<12 words)": 15.0,
    "short_penalty (<25 words)": 5.0,
}

# Filler words / sounds (whole-word / whole-phrase matches)
FILLER_PATTERNS = {
    "um": r"\bum+\b", "uh": r"\buh+\b", "er/erm": r"\berm?\b", "hmm": r"\bhm+\b",
    "basically": r"\bbasically\b", "actually": r"\bactually\b",
    "literally": r"\bliterally\b", "you know": r"\byou know\b",
    "i mean": r"\bi mean\b", "or something": r"\bor something\b",
    "and stuff": r"\band stuff\b",
    # 'like' is only a filler when it is set off by commas: ", like," - otherwise
    # it is a normal verb/preposition ("I like Python", "tools like Docker").
    "like (filler)": r"(?:^|,\s*)like\s*,",
}

# Hedging / uncertainty phrases
HEDGE_PATTERNS = {
    "i think": r"\bi think\b", "i guess": r"\bi guess\b", "i suppose": r"\bi suppose\b",
    "i feel like": r"\bi feel like\b", "maybe": r"\bmaybe\b", "perhaps": r"\bperhaps\b",
    "probably": r"\bprobably\b", "possibly": r"\bpossibly\b",
    "not sure": r"\bnot (?:totally |really |entirely )?sure\b",
    "don't know": r"\bdon'?t (?:really )?know\b", "sort of": r"\bsort of\b",
    "kind of": r"\bkind of\b", "somehow": r"\bsomehow\b",
    "not really": r"\b(?:not|don'?t|couldn'?t|can'?t) really\b",
    "something like": r"\bsomething like\b", "hoped": r"\bhop(?:e|ed)\b",
}

# Verbs that show ownership / concrete action
ACTION_VERBS = {
    "built", "designed", "implemented", "developed", "led", "optimized", "optimised",
    "reduced", "improved", "delivered", "architected", "created", "launched",
    "resolved", "deployed", "automated", "managed", "mentored", "analyzed",
    "analysed", "achieved", "increased", "migrated", "refactored", "owned",
    "diagnosed", "debugged", "benchmarked", "negotiated", "organized", "organised",
}


def _normalise(text: str) -> str:
    return str(text).lower().replace("\u2019", "'").replace("\u2018", "'")


def _count_patterns(patterns: dict, text: str) -> dict:
    hits = {}
    for name, pat in patterns.items():
        n = len(re.findall(pat, text))
        if n:
            hits[name] = n
    return hits


def _count_syllables(word: str) -> int:
    word = re.sub(r"e$", "", word)
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def fluency_metrics(raw_text: str) -> dict:
    text = _normalise(raw_text)
    words = re.findall(r"[a-z']+", text)
    n_words = max(len(words), 1)

    fillers = _count_patterns(FILLER_PATTERNS, text)
    hedges = _count_patterns(HEDGE_PATTERNS, text)
    n_fillers, n_hedges = sum(fillers.values()), sum(hedges.values())

    sentences = [s for s in re.split(r"[.!?\n]+", str(raw_text)) if s.strip()]
    n_sent = max(len(sentences), 1)

    repeats = len(re.findall(r"\b([a-z']+)\s+\1\b", text))      # "the the", "I I"
    actions = sum(1 for w in words if w in ACTION_VERBS)
    has_numbers = bool(re.search(r"\d", text))

    # Type-Token Ratio is meaningless on very short text, so we hide it.
    ttr = round(len(set(words)) / n_words * 100, 1) if len(words) >= 30 else None

    # Flesch Reading Ease (higher = easier). Syllables are estimated.
    syllables = sum(_count_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)

    return {
        "total_words": len(words),
        "sentences": n_sent,
        "avg_words_per_sentence": round(n_words / n_sent, 1),
        "filler_count": n_fillers,
        "filler_breakdown": fillers,
        "filler_per_100_words": round(n_fillers / n_words * 100, 2),
        "hedge_count": n_hedges,
        "hedge_breakdown": hedges,
        "hedge_per_100_words": round(n_hedges / n_words * 100, 2),
        "repeated_words": repeats,
        "action_verbs": actions,
        "has_numbers": has_numbers,
        "lexical_diversity_pct": ttr,
        "flesch_reading_ease": round(flesch, 1),
    }


def delivery_score(m: dict) -> dict:
    """Turn the fluency metrics into a 0-100 score + label + explanation."""
    W = SCORE_WEIGHTS
    parts = [("Base score", W["base"])]

    def add(label, value):
        if abs(value) > 1e-9:
            parts.append((label, round(value, 1)))

    add("Filler words", -min(35, W["filler_penalty_per_100_words"] * m["filler_per_100_words"]))
    add("Hedging phrases", -min(35, W["hedge_penalty_per_100_words"] * m["hedge_per_100_words"]))
    add("Repeated words (stutter)", -min(15, W["repeat_penalty_each"] * m["repeated_words"]))
    add("Action verbs (ownership)", min(15, W["action_verb_bonus_each"] * m["action_verbs"]))
    add("Mentions numbers / results", W["has_numbers_bonus"] if m["has_numbers"] else 0)
    if m["total_words"] < 12:
        add("Answer too short", -W["very_short_penalty (<12 words)"])
    elif m["total_words"] < 25:
        add("Answer fairly short", -W["short_penalty (<25 words)"])

    score = max(0.0, min(100.0, sum(v for _, v in parts)))
    label = "Confident" if score >= 70 else "Moderate" if score >= 45 else "Hesitant"
    return {"score": round(score, 1), "label": label, "breakdown": parts}


def feedback_tips(m: dict, relevance_pct=None, missing_terms=None) -> list:
    """Plain-English suggestions generated from the numbers above."""
    tips = []
    if m["filler_count"]:
        top = ", ".join(f"'{k}' x{v}" for k, v in sorted(m["filler_breakdown"].items(), key=lambda kv: -kv[1])[:3])
        tips.append(f"Cut filler words ({top}). Pause silently instead.")
    if m["hedge_count"]:
        top = ", ".join(f"'{k}'" for k in list(m["hedge_breakdown"])[:3])
        tips.append(f"Reduce hedging ({top}). State what you did directly, e.g. 'I implemented...'.")
    if m["repeated_words"]:
        tips.append("Repeated words detected - slow down and finish one thought before starting the next.")
    if m["action_verbs"] == 0 and m["total_words"] >= 12:
        tips.append("Use ownership verbs (built, designed, optimized, led) to show what YOU did.")
    if not m["has_numbers"] and m["total_words"] >= 12:
        tips.append("Add a measurable result (percent, time saved, users, size) to make the answer concrete.")
    if m["total_words"] < 25:
        tips.append("The answer is very short - add context, your action and the result.")
    if m["avg_words_per_sentence"] > 30:
        tips.append("Sentences are very long - break them up for clarity.")
    if relevance_pct is not None and relevance_pct < 40:
        tips.append("The answer looks off-topic for this question - address it directly in the first sentence.")
    if missing_terms:
        tips.append("Question terms not mentioned in the answer: " + ", ".join(missing_terms[:6]) + ".")
    if not tips:
        tips.append("Clear, direct and on-topic. Nothing major to fix.")
    return tips
