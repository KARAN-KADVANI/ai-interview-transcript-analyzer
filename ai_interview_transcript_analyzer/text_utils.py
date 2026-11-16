"""
text_utils.py
Shared text pre-processing used by BOTH training (train.py) and prediction
(evaluator.py). Keeping it in one file guarantees the model sees text cleaned
in exactly the same way at training time and at demo time.

No NLTK is used on purpose: NLTK needs a separate data download (stopwords /
wordnet) that often fails on college lab machines. Instead we use
  * scikit-learn's built-in English stop-word list
  * a small hand-written suffix-stripping stemmer (explained below)
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)

# (suffix, replacement) - tried from longest to shortest, first match wins.
_SUFFIX_RULES = [
    ("ization", "ize"), ("ational", "ate"), ("fulness", "ful"),
    ("ations", "ate"), ("ation", "ate"), ("ingly", ""), ("ments", ""),
    ("ment", ""), ("ness", ""), ("ities", "ity"), ("sses", "ss"),
    ("ies", "y"), ("ing", ""), ("ed", ""), ("ers", ""), ("er", ""),
    ("ly", ""), ("s", ""),
]


def stem(word: str) -> str:
    """Very light stemmer: servers/server/serving -> serv, queries/query -> query.
    The goal is only to merge obvious word variants so TF-IDF counts them
    together. It is not linguistically perfect (neither is Porter)."""
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        # keep at least 3 letters of the stem so "only" does not become "on"
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            w = word[: -len(suffix)] + replacement
            # undo doubled consonant: "running" -> "runn" -> "run"
            if replacement == "" and len(w) >= 3 and w[-1] == w[-2] and w[-1] not in "aeioulsz":
                w = w[:-1]
            word = w
            break
    # drop a trailing silent 'e' so cache/caching and database/databases agree
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def content_tokens(text: str):
    """lowercase -> letters only -> remove stop words -> stem. Returns a list."""
    text = str(text).lower().replace("\u2019", "'")
    words = re.findall(r"[a-z]+", text)
    return [stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]


def preprocess_text(text: str) -> str:
    """Same as content_tokens() but returns one space-joined string.
    This function is plugged into TfidfVectorizer(preprocessor=...)."""
    return " ".join(content_tokens(text))


def stem_to_word_map(text: str) -> dict:
    """stem -> first original word seen. Used only to show readable keywords
    in the UI (we display 'database', not the stem 'databas')."""
    mapping = {}
    for w in re.findall(r"[a-z]+", str(text).lower()):
        if w not in STOP_WORDS and len(w) > 1:
            mapping.setdefault(stem(w), w)
    return mapping
