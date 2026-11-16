"""
evaluator.py
Inference engine used by the Streamlit app. Loads the trained models saved by
train.py and analyses (a) one question/answer pair or (b) a whole interview
transcript with speaker labels.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np

import delivery
from text_utils import content_tokens, stem_to_word_map

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

_cache = {}


def models_ready() -> bool:
    return (MODEL_DIR / "category_model.pkl").exists() and (MODEL_DIR / "relevance_model.pkl").exists()


def load_models():
    if not models_ready():
        raise FileNotFoundError(
            "Trained models not found in 'models/'. Run  python train.py  first "
            "(after placing the Kaggle CSV in the 'data' folder).")
    if "category" not in _cache:
        _cache["category"] = joblib.load(MODEL_DIR / "category_model.pkl")
        _cache["relevance"] = joblib.load(MODEL_DIR / "relevance_model.pkl")
    return _cache["category"], _cache["relevance"]


def load_metrics():
    p = MODEL_DIR / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _relevance_label(pct):
    return "Relevant" if pct >= 70 else "Partially relevant" if pct >= 40 else "Off-topic"


def analyze_answer(question: str, transcript: str) -> dict:
    """Full analysis of one answer."""
    cat_model, rel_model = load_models()
    transcript = (transcript or "").strip()
    if len(content_tokens(transcript)) == 0:
        return {"error": "The transcript has no meaningful words to analyse."}

    # 1) Topic / category of what the candidate talked about (supervised ML)
    probs = cat_model.predict_proba([transcript])[0]
    order = np.argsort(probs)[::-1][:3]
    classes = cat_model.classes_
    top_categories = [(str(classes[i]), round(float(probs[i]) * 100, 1)) for i in order]

    # 2) Relevance to the question (supervised ML on similarity features)
    relevance_pct = None
    missing_terms = []
    if question and question.strip():
        relevance_pct = round(rel_model.score(question, transcript) * 100, 1)
        q_map = stem_to_word_map(question)
        a_stems = set(content_tokens(transcript))
        missing_terms = [w for s, w in q_map.items() if s not in a_stems]

    # 3) Delivery & fluency (rule-based)
    fluency = delivery.fluency_metrics(transcript)
    delivery_res = delivery.delivery_score(fluency)

    # 4) Keywords the candidate used (highest TF-IDF weight under the category model)
    keywords = _top_keywords(cat_model, transcript)

    return {
        "top_categories": top_categories,
        "relevance_pct": relevance_pct,
        "relevance_label": _relevance_label(relevance_pct) if relevance_pct is not None else None,
        "delivery": delivery_res,
        "fluency": fluency,
        "keywords": keywords,
        "missing_question_terms": missing_terms,
        "tips": delivery.feedback_tips(fluency, relevance_pct, missing_terms),
    }


def _top_keywords(cat_model, transcript, k=6):
    tfidf = cat_model.named_steps["tfidf"]
    vec = tfidf.transform([transcript])
    names = tfidf.get_feature_names_out()
    row = vec.toarray()[0]
    stem_map = stem_to_word_map(transcript)
    out = []
    for i in np.argsort(row)[::-1]:
        if row[i] <= 0 or len(out) >= k:
            break
        term = names[i]
        if " " in term:                       # skip bigrams for readability
            continue
        out.append(stem_map.get(term, term))
    return out


# --------------------------------------------------------------------------
# Full-transcript support
# --------------------------------------------------------------------------
_Q_RE = re.compile(r"^\s*(interviewer|recruiter|hr|panel(?:ist)?|question|q)\s*[:\-]\s*(.*)$", re.I)
_A_RE = re.compile(r"^\s*(candidate|interviewee|applicant|answer|a)\s*[:\-]\s*(.*)$", re.I)


def parse_transcript(text: str):
    """Split a labelled transcript into [(question, answer), ...].
    Lines look like  'Interviewer: ...'  /  'Candidate: ...'  (Q:/A: also work).
    Lines without a label are treated as a continuation of the previous speaker."""
    turns = []                      # list of [role, text]
    for line in str(text).splitlines():
        if not line.strip():
            continue
        mq, ma = _Q_RE.match(line), _A_RE.match(line)
        if mq:
            turns.append(["Q", mq.group(2).strip()])
        elif ma:
            turns.append(["A", ma.group(2).strip()])
        elif turns:
            turns[-1][1] += " " + line.strip()
    pairs, current_q = [], None
    for role, t in turns:
        if role == "Q":
            current_q = t
        elif role == "A" and current_q is not None:
            if pairs and pairs[-1][0] == current_q:       # two answers to one question
                pairs[-1] = (current_q, pairs[-1][1] + " " + t)
            else:
                pairs.append((current_q, t))
    return pairs


def analyze_interview(text: str) -> dict:
    pairs = parse_transcript(text)
    if not pairs:
        return {"error": "No Interviewer/Candidate turns found. Start lines with "
                         "'Interviewer:' and 'Candidate:' (or 'Q:' and 'A:')."}
    rows = []
    for i, (q, a) in enumerate(pairs, 1):
        res = analyze_answer(q, a)
        if "error" in res:
            continue
        rows.append({"n": i, "question": q, "answer": a, "result": res})
    if not rows:
        return {"error": "Could not analyse any answers in this transcript."}
    rel = [r["result"]["relevance_pct"] for r in rows if r["result"]["relevance_pct"] is not None]
    summary = {
        "answers_analysed": len(rows),
        "avg_relevance_pct": round(float(np.mean(rel)), 1) if rel else None,
        "avg_delivery_score": round(float(np.mean([r["result"]["delivery"]["score"] for r in rows])), 1),
        "total_fillers": int(sum(r["result"]["fluency"]["filler_count"] for r in rows)),
        "total_hedges": int(sum(r["result"]["fluency"]["hedge_count"] for r in rows)),
    }
    return {"rows": rows, "summary": summary}
