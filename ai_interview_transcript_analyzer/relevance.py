"""
relevance.py
Answer-relevance model: "does this answer actually address this question?"

Instead of a single raw cosine similarity, we train a small Logistic
Regression on FIVE similarity features computed between a question and an
answer:

  1. word_cosine   TF-IDF (1-2 grams, stemmed words) cosine similarity
  2. char_cosine   TF-IDF over character 3-5 grams (robust to spelling /
                   word-form differences, e.g. "indexing" vs "indexes")
  3. lsa_cosine    cosine in a Latent Semantic Analysis space (TruncatedSVD of
                   the TF-IDF matrix) - catches related-but-not-identical words
  4. coverage      fraction of the question's content words found in the answer
  5. jaccard       word-set overlap between question and answer

Training data (all built from the REAL Kaggle question/answer pairs):
  positive pair  = (question_i , its own answer_i)               label 1
  negative pair  = (question_i , answer of a different question) label 0
                   one random negative + one "hard" negative taken from the
                   SAME category (same topic, wrong answer) per question.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from text_utils import content_tokens, preprocess_text

FEATURE_NAMES = ["word_cosine", "char_cosine", "lsa_cosine", "coverage", "jaccard"]


def _row_dot(A, B):
    """Row-wise dot product of two L2-normalised sparse matrices = cosine."""
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


class RelevanceModel:
    def __init__(self, n_lsa: int = 50, seed: int = 42):
        self.n_lsa = n_lsa
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, questions, answers, categories):
        questions, answers = list(questions), list(answers)
        corpus = questions + answers

        self.word_vec = TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2),
                                        sublinear_tf=True).fit(corpus)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        sublinear_tf=True).fit(corpus)
        word_matrix = self.word_vec.transform(corpus)
        n_comp = max(2, min(self.n_lsa, word_matrix.shape[0] - 1, word_matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed).fit(word_matrix)

        qs, ans, y = self._build_pairs(questions, answers, categories)
        X = self.features(qs, ans)
        self.scaler = StandardScaler().fit(X)
        # class_weight='balanced': we have 1 positive : 2 negatives, this keeps
        # the 0.5 probability threshold meaningful.
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        self.clf.fit(self.scaler.transform(X), y)
        return self

    def _build_pairs(self, questions, answers, categories):
        rng = np.random.default_rng(self.seed)
        n = len(questions)
        categories = np.asarray(categories)
        qs, ans, y = [], [], []
        for i in range(n):
            qs.append(questions[i]); ans.append(answers[i]); y.append(1)
            # random negative
            j = int(rng.integers(n))
            while j == i:
                j = int(rng.integers(n))
            qs.append(questions[i]); ans.append(answers[j]); y.append(0)
            # hard negative (same category, different row)
            same = np.where((categories == categories[i]) & (np.arange(n) != i))[0]
            if len(same):
                j = int(rng.choice(same))
                qs.append(questions[i]); ans.append(answers[j]); y.append(0)
        return qs, ans, np.array(y)

    # ------------------------------------------------------------- features
    def features(self, qs, ans):
        qs, ans = list(qs), list(ans)
        Qw, Aw = self.word_vec.transform(qs), self.word_vec.transform(ans)
        word_cos = _row_dot(Qw, Aw)

        Qc = self.char_vec.transform([q.lower() for q in qs])
        Ac = self.char_vec.transform([a.lower() for a in ans])
        char_cos = _row_dot(Qc, Ac)

        Qs, As = self.svd.transform(Qw), self.svd.transform(Aw)
        Qs /= np.linalg.norm(Qs, axis=1, keepdims=True) + 1e-9
        As /= np.linalg.norm(As, axis=1, keepdims=True) + 1e-9
        lsa_cos = (Qs * As).sum(axis=1)

        coverage, jaccard = [], []
        for q, a in zip(qs, ans):
            qt, at = set(content_tokens(q)), set(content_tokens(a))
            coverage.append(len(qt & at) / len(qt) if qt else 0.0)
            jaccard.append(len(qt & at) / len(qt | at) if (qt | at) else 0.0)
        return np.column_stack([word_cos, char_cos, lsa_cos, coverage, jaccard])

    # -------------------------------------------------------------- predict
    def predict_proba(self, qs, ans):
        """Probability (0-1) that each answer belongs to its question."""
        X = self.scaler.transform(self.features(qs, ans))
        return self.clf.predict_proba(X)[:, 1]

    def score(self, question: str, answer: str) -> float:
        return float(self.predict_proba([question], [answer])[0])

    def similarity_matrix(self, questions, answers):
        """S[i, j] = P(answer j matches question i). Used for retrieval metrics."""
        n_q, n_a = len(questions), len(answers)
        qs = [questions[i] for i in range(n_q) for _ in range(n_a)]
        an = [answers[j] for _ in range(n_q) for j in range(n_a)]
        return self.predict_proba(qs, an).reshape(n_q, n_a)

    def feature_weights(self):
        return dict(zip(FEATURE_NAMES, [round(float(w), 3) for w in self.clf.coef_[0]]))
