"""
data_loader.py
Loads the Kaggle CSV and converts it into a clean DataFrame with the standard
columns: question, answer, category (and difficulty if present).

Dataset used:  "Software Engineering Interview Questions Dataset"
Kaggle author: syedmharis
File:          Software Questions.csv
Columns:       Question Number | Question | Brief Answer | Category | Difficulty

The loader finds columns by name pattern (not exact spelling), so slightly
different CSVs (e.g. another Kaggle interview Q&A dataset with 'question',
'answer', 'topic') also work without code changes.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HELP_MESSAGE = """
Could not find the dataset CSV.

1. Open https://www.kaggle.com/datasets/syedmharis/software-engineering-interview-questions-dataset
2. Click Download and unzip it.
3. Copy 'Software Questions.csv' into the 'data' folder of this project
   (or run:  python train.py --data "path/to/your.csv")
"""


def _find_col(columns, include, exclude=()):
    for c in columns:
        name = c.lower().strip()
        if any(k in name for k in include) and not any(k in name for k in exclude):
            return c
    return None


def _read_csv_any_encoding(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def resolve_path(path=None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}\n{HELP_MESSAGE}")
        return p
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(HELP_MESSAGE)
    return csvs[0]


def load_dataset(path=None, min_per_class: int = 6, verbose: bool = True):
    """Return (clean_df, info_dict)."""
    csv_path = resolve_path(path)
    raw = _read_csv_any_encoding(csv_path)
    cols = list(raw.columns)

    q_col = _find_col(cols, ["question"], exclude=["number", "no", "id"])
    a_col = _find_col(cols, ["answer", "response", "solution"])
    c_col = _find_col(cols, ["category", "topic", "domain", "type"])
    d_col = _find_col(cols, ["difficult", "level"])

    missing = [n for n, c in [("question", q_col), ("answer", a_col), ("category", c_col)] if c is None]
    if missing:
        raise ValueError(f"Could not find column(s) {missing} in {csv_path.name}. "
                         f"Columns present: {cols}")

    df = pd.DataFrame({"question": raw[q_col], "answer": raw[a_col], "category": raw[c_col]})
    if d_col:
        df["difficulty"] = raw[d_col]

    n_raw = len(df)
    df = df.dropna(subset=["question", "answer", "category"])
    for c in ["question", "answer", "category"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question"])
    df = df[df["answer"].str.split().str.len() >= 3]          # drop empty / 1-2 word answers

    # a class needs enough rows to be split into train/test and cross-validated
    counts = df["category"].value_counts()
    rare = counts[counts < min_per_class].index.tolist()
    if rare:
        if verbose:
            print(f"[data] Dropping rare categories (< {min_per_class} rows): {rare}")
        df = df[~df["category"].isin(rare)]
    df = df.reset_index(drop=True)

    info = {
        "file": csv_path.name,
        "rows_raw": int(n_raw),
        "rows_used": int(len(df)),
        "n_categories": int(df["category"].nunique()),
        "categories": df["category"].value_counts().to_dict(),
        "avg_answer_words": round(float(df["answer"].str.split().str.len().mean()), 1),
        "avg_question_words": round(float(df["question"].str.split().str.len().mean()), 1),
        "columns_detected": {"question": q_col, "answer": a_col, "category": c_col, "difficulty": d_col},
    }
    if verbose:
        print(f"[data] Loaded '{csv_path.name}': {n_raw} raw rows -> {len(df)} usable rows, "
              f"{info['n_categories']} categories")
    return df, info
