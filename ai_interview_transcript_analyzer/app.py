"""
app.py - Streamlit demo for the AI Interview Transcript Analyzer.
Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import evaluator
from samples import FULL_TRANSCRIPT_SAMPLE, SINGLE_SAMPLES

st.set_page_config(page_title="AI Interview Transcript Analyzer", layout="wide")
st.title("🎙️ AI Interview Transcript Analyzer")
st.caption("Classical NLP + machine learning (TF-IDF, SVM / Logistic Regression, LSA). "
           "No LLMs and no external APIs - everything runs locally.")

if not evaluator.models_ready():
    st.error("Trained models not found. Put the Kaggle CSV in the `data` folder and run "
             "`python train.py` once, then restart this app.")
    st.stop()

# ------------------------------------------------------------------ sidebar
metrics = evaluator.load_metrics()
with st.sidebar:
    st.header("Model information")
    if metrics:
        d, c, r = metrics["dataset"], metrics["category_model"], metrics["relevance_model"]
        st.write(f"**Dataset:** {d['file']}  \n{d['rows_used']} Q&A pairs, {d['n_categories']} categories")
        st.write(f"**Category model:** {c['best_model']}")
        st.metric("Category accuracy (held-out test)", f"{c['test_accuracy'] * 100:.1f}%")
        st.metric("Category macro-F1", f"{c['test_macro_f1'] * 100:.1f}%")
        st.metric("Relevance ROC-AUC", f"{r['test_roc_auc']:.3f}")
        st.metric("Question retrieval top-1", f"{r['retrieval_top1'] * 100:.1f}%")
    st.markdown("---")
    st.caption("Delivery / confidence score is **rule-based** (filler words, hedges, "
               "repeats, action verbs, numbers), not a trained model.")


def show_result(res: dict):
    if "error" in res:
        st.warning(res["error"])
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected topic", res["top_categories"][0][0])
    if res["relevance_pct"] is not None:
        m2.metric("Relevance to question", f"{res['relevance_pct']}%")
        m2.caption(res["relevance_label"])
    m3.metric("Delivery score", f"{res['delivery']['score']}/100")
    m3.caption(res["delivery"]["label"])
    m4.metric("Filler words", res["fluency"]["filler_count"])

    t1, t2, t3, t4 = st.tabs(["Feedback", "Topic prediction", "Delivery details", "Raw metrics"])
    with t1:
        for tip in res["tips"]:
            st.write("• " + tip)
        if res["keywords"]:
            st.write("**Key terms you used:** " + ", ".join(res["keywords"]))
    with t2:
        st.write("Top-3 categories predicted by the ML model:")
        st.bar_chart(pd.Series({k: v for k, v in res["top_categories"]}, name="probability %"))
    with t3:
        st.write("How the delivery score was computed:")
        st.table(pd.DataFrame(res["delivery"]["breakdown"], columns=["Factor", "Points"]))
        f = res["fluency"]
        if f["filler_breakdown"]:
            st.write("**Filler words found:**")
            st.bar_chart(pd.Series(f["filler_breakdown"]))
        if f["hedge_breakdown"]:
            st.write("**Hedging phrases found:**")
            st.bar_chart(pd.Series(f["hedge_breakdown"]))
    with t4:
        st.json(res["fluency"])


tab_single, tab_full = st.tabs(["Single question & answer", "Full interview transcript"])

with tab_single:
    choice = st.radio("Load a demo example or type your own:",
                      ["Custom"] + list(SINGLE_SAMPLES), horizontal=True)
    q0, a0 = SINGLE_SAMPLES.get(choice, ("Explain the difference between a process and a thread.", ""))
    question = st.text_area("Interview question", value=q0, key=f"q_{choice}", height=80)
    answer = st.text_area("Candidate answer (transcript)", value=a0, key=f"a_{choice}", height=170)
    if st.button("Analyze answer", type="primary"):
        show_result(evaluator.analyze_answer(question, answer))

with tab_full:
    st.write("Paste a full transcript. Start each turn with `Interviewer:` and `Candidate:` "
             "(`Q:` / `A:` also work).")
    text = st.text_area("Full transcript", value=FULL_TRANSCRIPT_SAMPLE, height=260, key="full_text")
    if st.button("Analyze full interview", type="primary"):
        out = evaluator.analyze_interview(text)
        if "error" in out:
            st.warning(out["error"])
        else:
            s = out["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Answers analysed", s["answers_analysed"])
            c2.metric("Avg relevance", f"{s['avg_relevance_pct']}%")
            c3.metric("Avg delivery score", s["avg_delivery_score"])
            c4.metric("Fillers + hedges", s["total_fillers"] + s["total_hedges"])
            table = pd.DataFrame([{
                "#": r["n"],
                "Question": r["question"][:70],
                "Topic": r["result"]["top_categories"][0][0],
                "Relevance %": r["result"]["relevance_pct"],
                "Delivery": r["result"]["delivery"]["score"],
                "Tone": r["result"]["delivery"]["label"],
                "Fillers": r["result"]["fluency"]["filler_count"],
                "Hedges": r["result"]["fluency"]["hedge_count"],
            } for r in out["rows"]])
            st.dataframe(table, hide_index=True)
            for r in out["rows"]:
                with st.expander(f"Answer {r['n']}: {r['question'][:80]}"):
                    show_result(r["result"])
