"""
train.py - trains and evaluates every ML component on the REAL Kaggle dataset.

Run:   python train.py                       (uses the first CSV found in data/)
       python train.py --data "path/to.csv"

What it does
  1. Loads + cleans the Kaggle CSV, prints dataset statistics, saves a chart.
  2. Splits the data 80/20 (stratified by category) ONCE. The test set is never
     used for training or model selection.
  3. CATEGORY MODEL: compares Majority-baseline, Naive Bayes, Logistic
     Regression and a calibrated Linear SVM with 5-fold cross-validation on the
     training part, picks the best by macro-F1, evaluates it on the test set.
  4. RELEVANCE MODEL: trains on question/answer pairs from the training rows,
     evaluates on pairs built ONLY from unseen test rows (ROC-AUC, F1 and a
     retrieval test: can the model find the right question for an answer?).
  5. Re-trains both final models on 100% of the data and saves them in models/.
  6. Writes metrics.json, charts and a text report (used for the PPT/report).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_dataset
from relevance import RelevanceModel
from text_utils import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR, REPORT_DIR = BASE_DIR / "models", BASE_DIR / "reports"
SEED = 42


def make_pipeline(clf):
    """Raw text -> (clean + stem) -> TF-IDF 1-2 grams -> classifier.
    Everything lives inside ONE Pipeline, so inside cross-validation the
    vectoriser is re-fitted on each training fold only (no leakage)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ("clf", clf),
    ])


def retrieval_metrics(S):
    """S[i, j] = score of (question i, answer j). For every answer j we rank all
    questions; the correct one is question j."""
    n = S.shape[0]
    ranks = np.array([1 + int((S[:, j] > S[j, j]).sum()) for j in range(n)])
    return {"top1": float((ranks == 1).mean()), "top3": float((ranks <= 3).mean()),
            "mrr": float((1.0 / ranks).mean()), "random_top1": 1.0 / n}


def main(data_path):
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------ 1. data
    print("\n=== 1. DATA ===")
    df, info = load_dataset(data_path)
    print(df["category"].value_counts().to_string())
    print(f"Average answer length: {info['avg_answer_words']} words")

    counts = df["category"].value_counts().sort_values()
    plt.figure(figsize=(8, 5))
    plt.barh(counts.index, counts.values, color="#2b6cb0")
    plt.xlabel("Number of Q&A pairs"); plt.title("Class distribution")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150); plt.close()

    # --------------------------------------------------- 2. one split only
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2,
                                           stratify=df["category"], random_state=SEED)
    tr, te = df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)
    print(f"\nTrain rows: {len(tr)}   Test rows: {len(te)}")

    # ------------------------------------------------ 3. category model
    print("\n=== 2. CATEGORY MODEL (answer text -> topic) ===")
    min_cnt = int(tr["category"].value_counts().min())
    n_splits = max(2, min(5, min_cnt))
    cal_cv = max(2, min(3, min_cnt))
    candidates = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=10, class_weight="balanced", max_iter=2000),
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), cv=cal_cv, method="sigmoid"),
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_results = {}
    for name, clf in candidates.items():
        cv = cross_validate(make_pipeline(clf), tr["answer"], tr["category"], cv=skf,
                            scoring=["accuracy", "f1_macro"])
        cv_results[name] = {"cv_accuracy_mean": float(cv["test_accuracy"].mean()),
                            "cv_accuracy_std": float(cv["test_accuracy"].std()),
                            "cv_macro_f1_mean": float(cv["test_f1_macro"].mean())}
        print(f"{name:26s} CV accuracy {cv_results[name]['cv_accuracy_mean']*100:5.1f}% "
              f"(+/- {cv_results[name]['cv_accuracy_std']*100:.1f})   "
              f"CV macro-F1 {cv_results[name]['cv_macro_f1_mean']*100:5.1f}%")

    real_models = [n for n in candidates if n != "Majority baseline"]
    best_name = max(real_models, key=lambda n: cv_results[n]["cv_macro_f1_mean"])
    print(f"\nBest model by cross-validated macro-F1: {best_name}")

    # fair test-set comparison (all models trained on train split only)
    test_scores = {}
    for name, clf in candidates.items():
        p = make_pipeline(clf).fit(tr["answer"], tr["category"])
        test_scores[name] = float(accuracy_score(te["category"], p.predict(te["answer"])))
    best_pipe = make_pipeline(candidates[best_name]).fit(tr["answer"], tr["category"])
    y_pred = best_pipe.predict(te["answer"])
    test_acc = float(accuracy_score(te["category"], y_pred))
    test_f1 = float(f1_score(te["category"], y_pred, average="macro", zero_division=0))
    test_wf1 = float(f1_score(te["category"], y_pred, average="weighted", zero_division=0))
    report_txt = classification_report(te["category"], y_pred, zero_division=0)
    print(f"\nHELD-OUT TEST  accuracy {test_acc*100:.1f}%   macro-F1 {test_f1*100:.1f}%   "
          f"(majority baseline {test_scores['Majority baseline']*100:.1f}%)")
    print(report_txt)

    labels = sorted(df["category"].unique())
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(te["category"], y_pred, labels=labels, ax=ax,
                                            xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name} (test set)")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    names = list(cv_results)
    plt.bar(names, [cv_results[n]["cv_accuracy_mean"] * 100 for n in names],
            yerr=[cv_results[n]["cv_accuracy_std"] * 100 for n in names], color="#2b6cb0", capsize=4)
    plt.ylabel("CV accuracy (%)"); plt.xticks(rotation=15); plt.title("Category model comparison (5-fold CV)")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150); plt.close()

    # what did the model learn? top words per class from a plain LR
    vec = TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2), sublinear_tf=True)
    Xtr = vec.fit_transform(tr["answer"])
    lr = LogisticRegression(C=10, max_iter=2000).fit(Xtr, tr["category"])
    names_arr = np.array(vec.get_feature_names_out())
    with open(REPORT_DIR / "top_terms_per_category.txt", "w", encoding="utf-8") as f:
        for ci, cname in enumerate(lr.classes_):
            top = names_arr[np.argsort(lr.coef_[ci])[::-1][:8]]
            f.write(f"{cname}: {', '.join(top)}\n")

    # ------------------------------------------------ 4. relevance model
    print("=== 3. RELEVANCE MODEL (question + answer -> does it match?) ===")
    rel = RelevanceModel(seed=SEED).fit(tr["question"], tr["answer"], tr["category"])
    # pairs from UNSEEN test rows only
    t_q, t_a, t_y = rel._build_pairs(list(te["question"]), list(te["answer"]), te["category"])
    prob = rel.predict_proba(t_q, t_a)
    pred = (prob >= 0.5).astype(int)
    rel_metrics = {
        "test_roc_auc": float(roc_auc_score(t_y, prob)),
        "test_accuracy": float(accuracy_score(t_y, pred)),
        "test_precision": float(precision_score(t_y, pred, zero_division=0)),
        "test_recall": float(recall_score(t_y, pred, zero_division=0)),
        "test_f1": float(f1_score(t_y, pred, zero_division=0)),
        "baseline_word_cosine_auc": float(roc_auc_score(t_y, rel.features(t_q, t_a)[:, 0])),
        "n_test_pairs": int(len(t_y)),
        "feature_weights": rel.feature_weights(),
    }
    S = rel.similarity_matrix(list(te["question"]), list(te["answer"]))
    ret = retrieval_metrics(S)
    # baseline: plain TF-IDF cosine only
    Qw = rel.word_vec.transform(list(te["question"])); Aw = rel.word_vec.transform(list(te["answer"]))
    base = retrieval_metrics((Qw @ Aw.T).toarray())
    rel_metrics.update({"retrieval_top1": ret["top1"], "retrieval_top3": ret["top3"],
                        "retrieval_mrr": ret["mrr"], "retrieval_random_top1": ret["random_top1"],
                        "baseline_cosine_top1": base["top1"], "baseline_cosine_top3": base["top3"],
                        "baseline_cosine_mrr": base["mrr"], "n_test_questions": int(len(te))})
    print(f"Pair classification  AUC {rel_metrics['test_roc_auc']:.3f} "
          f"(plain cosine AUC {rel_metrics['baseline_word_cosine_auc']:.3f})  "
          f"F1 {rel_metrics['test_f1']:.3f}  accuracy {rel_metrics['test_accuracy']*100:.1f}%")
    print(f"Retrieval among {len(te)} unseen questions: top-1 {ret['top1']*100:.1f}%  "
          f"top-3 {ret['top3']*100:.1f}%  MRR {ret['mrr']:.3f}  (random top-1 {ret['random_top1']*100:.1f}%)")
    print(f"   plain cosine baseline: top-1 {base['top1']*100:.1f}%  top-3 {base['top3']*100:.1f}%  MRR {base['mrr']:.3f}")
    print("Learned feature weights:", rel_metrics["feature_weights"])

    # --------------------------------------- 5. final models on all data
    print("\n=== 4. SAVING FINAL MODELS (re-trained on 100% of the data) ===")
    final_cat = make_pipeline(candidates[best_name]).fit(df["answer"], df["category"])
    final_rel = RelevanceModel(seed=SEED).fit(df["question"], df["answer"], df["category"])
    joblib.dump(final_cat, MODEL_DIR / "category_model.pkl")
    joblib.dump(final_rel, MODEL_DIR / "relevance_model.pkl")

    metrics = {
        "dataset": info,
        "split": {"train_rows": int(len(tr)), "test_rows": int(len(te)), "seed": SEED, "cv_folds": n_splits},
        "category_model": {"best_model": best_name, "cv_results": cv_results,
                           "test_accuracy": test_acc, "test_macro_f1": test_f1,
                           "test_weighted_f1": test_wf1,
                           "test_accuracy_all_models": test_scores,
                           "baseline_accuracy": test_scores["Majority baseline"]},
        "relevance_model": rel_metrics,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    (REPORT_DIR / "results_summary.md").write_text(_summary_md(metrics), encoding="utf-8")
    print("Saved: models/*.pkl, models/metrics.json, reports/*.png, reports/*.txt, reports/results_summary.md")
    print("\nDone. Start the demo with:  streamlit run app.py")


def _summary_md(m):
    c, r, d = m["category_model"], m["relevance_model"], m["dataset"]
    lines = [f"# Results summary", "",
             f"Dataset: {d['file']} - {d['rows_used']} usable rows, {d['n_categories']} categories", "",
             "## Category model (held-out test set)", "",
             "| Model | CV accuracy | Test accuracy |", "|---|---|---|"]
    for n, v in c["cv_results"].items():
        lines.append(f"| {n} | {v['cv_accuracy_mean']*100:.1f}% | {c['test_accuracy_all_models'][n]*100:.1f}% |")
    lines += ["", f"Best model: **{c['best_model']}** - accuracy {c['test_accuracy']*100:.1f}%, "
              f"macro-F1 {c['test_macro_f1']*100:.1f}%", "",
              "## Relevance model (unseen test questions)", "",
              f"- ROC-AUC {r['test_roc_auc']:.3f} (plain TF-IDF cosine: {r['baseline_word_cosine_auc']:.3f})",
              f"- F1 {r['test_f1']:.3f}, accuracy {r['test_accuracy']*100:.1f}%",
              f"- Retrieval top-1 {r['retrieval_top1']*100:.1f}%, top-3 {r['retrieval_top3']*100:.1f}%, "
              f"MRR {r['retrieval_mrr']:.3f} (random top-1: {r['retrieval_random_top1']*100:.1f}%)",
              f"- Plain-cosine retrieval: top-1 {r['baseline_cosine_top1']*100:.1f}%, "
              f"top-3 {r['baseline_cosine_top3']*100:.1f}%, MRR {r['baseline_cosine_mrr']:.3f}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to the Kaggle CSV (default: first CSV in data/)")
    main(ap.parse_args().data)
