# Results summary

Dataset: Software Questions.csv - 147 usable rows, 13 categories

## Category model (held-out test set)

| Model | CV accuracy | Test accuracy |
|---|---|---|
| Majority baseline | 16.2% | 16.7% |
| Naive Bayes | 31.7% | 36.7% |
| Logistic Regression | 57.3% | 50.0% |
| Linear SVM (calibrated) | 58.2% | 53.3% |

Best model: **Linear SVM (calibrated)** - accuracy 53.3%, macro-F1 41.8%

## Relevance model (unseen test questions)

- ROC-AUC 0.886 (plain TF-IDF cosine: 0.788)
- F1 0.700, accuracy 79.8%
- Retrieval top-1 56.7%, top-3 80.0%, MRR 0.703 (random top-1: 3.3%)
- Plain-cosine retrieval: top-1 53.3%, top-3 86.7%, MRR 0.709
