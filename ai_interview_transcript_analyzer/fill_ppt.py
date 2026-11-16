"""
fill_ppt.py - puts YOUR real results into the presentation.

The PPT ships with tokens such as {{CAT_ACC}}. After you run  python train.py
this script reads models/metrics.json and replaces every token with the
measured number, and inserts reports/confusion_matrix.png and
reports/class_distribution.png into the reserved spots.

Usage (run from the project folder):
  pip install python-pptx
  python fill_ppt.py "AI_Interview_Transcript_Analyzer.pptx" ^
        --github https://github.com/<user>/<repo> ^
        --team "Name 1, Name 2, Name 3" --guide "Prof. Name"
Output: <name>_filled.pptx   (the original is not modified)
"""
import argparse
import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation

BASE = Path(__file__).resolve().parent


def pct(x):
    return f"{x * 100:.1f}%"


def build_tokens(m, args):
    d, c, r, sp = m["dataset"], m["category_model"], m["relevance_model"], m["split"]
    cv, te = c["cv_results"], c["test_accuracy_all_models"]
    t = {
        "N_ROWS": str(d["rows_used"]), "N_CATS": str(d["n_categories"]),
        "AVG_WORDS": str(d["avg_answer_words"]),
        "N_TRAIN": str(sp["train_rows"]), "N_TEST": str(sp["test_rows"]),
        "CAT_BEST": c["best_model"].replace(" (calibrated)", ""),
        "CAT_ACC": pct(c["test_accuracy"]), "CAT_F1": pct(c["test_macro_f1"]),
        "REL_AUC": f"{r['test_roc_auc']:.3f}", "REL_AUC_BASE": f"{r['baseline_word_cosine_auc']:.3f}",
        "TOP1": pct(r["retrieval_top1"]), "TOP1_BASE": pct(r["baseline_cosine_top1"]),
        "RANDOM_TOP1": pct(r["retrieval_random_top1"]),
        "TOP3": pct(r["retrieval_top3"]), "TOP3_BASE": pct(r["baseline_cosine_top3"]),
        "MRR": f"{r['retrieval_mrr']:.3f}", "MRR_BASE": f"{r['baseline_cosine_mrr']:.3f}",
    }
    for key, name in [("BASE", "Majority baseline"), ("NB", "Naive Bayes"),
                      ("LR", "Logistic Regression"), ("SVM", "Linear SVM (calibrated)")]:
        t[f"CV_{key}"] = pct(cv[name]["cv_accuracy_mean"])
        t[f"TEST_{key}"] = pct(te[name])
    if args.github: t["GITHUB_URL"] = args.github
    if args.team: t["TEAM"] = args.team
    if args.guide: t["GUIDE"] = args.guide
    return t


def walk(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:          # group
            yield from walk(sh.shapes)


def replace_in_frame(tf, tokens):
    for p in tf.paragraphs:
        for run in p.runs:
            if "{{" in run.text:
                run.text = re.sub(r"\{\{(\w+)\}\}",
                                  lambda mo: tokens.get(mo.group(1), mo.group(0)), run.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx")
    ap.add_argument("--metrics", default=str(BASE / "models" / "metrics.json"))
    ap.add_argument("--reports", default=str(BASE / "reports"))
    ap.add_argument("--github"); ap.add_argument("--team"); ap.add_argument("--guide")
    ap.add_argument("--out")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    tokens = build_tokens(metrics, args)
    prs = Presentation(args.pptx)
    images = {"CM_IMAGE": Path(args.reports) / "confusion_matrix.png",
              "CLASSDIST_IMAGE": Path(args.reports) / "class_distribution.png"}
    hints = {"cmHint", "classdistHint"}

    for slide in prs.slides:
        for sh in list(walk(slide.shapes)):
            if sh.name in hints:
                sh._element.getparent().remove(sh._element)
            elif sh.name in images and images[sh.name].exists():
                l, t, w, h = sh.left, sh.top, sh.width, sh.height
                iw, ih = Image.open(images[sh.name]).size
                scale = min(w / iw, h / ih)
                pw, ph = int(iw * scale), int(ih * scale)
                sh._element.getparent().remove(sh._element)
                slide.shapes.add_picture(str(images[sh.name]), l + (w - pw) // 2, t + (h - ph) // 2, pw, ph)
            elif sh.has_text_frame:
                replace_in_frame(sh.text_frame, tokens)
            elif sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        replace_in_frame(cell.text_frame, tokens)

    out = args.out or str(Path(args.pptx).with_name(Path(args.pptx).stem + "_filled.pptx"))
    prs.save(out)
    left = set()
    for slide in Presentation(out).slides:
        for sh in walk(slide.shapes):
            texts = []
            if sh.has_text_frame: texts.append(sh.text_frame.text)
            if sh.has_table: texts += [c.text for r in sh.table.rows for c in r.cells]
            for tx in texts: left.update(re.findall(r"\{\{(\w+)\}\}", tx))
    print(f"Saved {out}")
    if left:
        print("Still unfilled (pass the matching option or edit in PowerPoint):", ", ".join(sorted(left)))


if __name__ == "__main__":
    main()
