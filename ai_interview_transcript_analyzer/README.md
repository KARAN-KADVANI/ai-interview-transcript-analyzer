# AI Interview Transcript Analyzer

Analyses interview answers with **classical NLP + simple ML only** - no LLMs, no external APIs.

| Module | What it does | How |
|---|---|---|
| Topic detection | Which topic is the candidate talking about? | TF-IDF + Logistic Regression / SVM / Naive Bayes (best chosen by cross-validation) |
| Relevance | Does the answer address the question? | Logistic Regression on 5 similarity features (word TF-IDF, char TF-IDF, LSA, coverage, Jaccard) |
| Delivery / confidence | Fillers, hedging, repeats, action verbs, numbers | Transparent rule-based scorer (explained in `delivery.py`) |
| Full-transcript mode | Splits `Interviewer:` / `Candidate:` turns and scores each answer | Rule-based parser + the models above |

## Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

1. Download the dataset from Kaggle (link in `data/README.md`) and copy
   `Software Questions.csv` into the `data/` folder.
2. Train + evaluate (about 1 minute):
   ```bash
   python train.py
   ```
   This prints all metrics and writes `models/` and `reports/` (charts, report).
3. Start the demo:
   ```bash
   streamlit run app.py
   ```

## Files

- `data_loader.py` - reads/cleans the Kaggle CSV (auto-detects columns)
- `text_utils.py` - tokenising, stop-words, stemmer (shared by training and demo)
- `relevance.py` - relevance model class
- `delivery.py` - filler/hedge/fluency analysis and delivery score
- `train.py` - trains, cross-validates, evaluates, saves models and reports
- `evaluator.py` - loads models, analyses an answer or a whole transcript
- `samples.py` - demo examples for the app
- `app.py` - Streamlit user interface
- `fill_ppt.py` - fills the presentation with your measured results

## Presentation

`fill_ppt.py` writes your real results (from `models/metrics.json`) and charts into the PPT:

```bash
pip install python-pptx
python fill_ppt.py "AI_Interview_Transcript_Analyzer.pptx" --github https://github.com/<user>/<repo> --team "Name1, Name2" --guide "Prof. Name"
```

Full explanation of the project: see `PROJECT_GUIDE` (provided separately).
