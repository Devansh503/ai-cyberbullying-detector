import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources are available
for pkg in ["punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg)
        except Exception:
            pass

STOPWORDS = set(stopwords.words('english')) if 'english' in stopwords.fileids() else set()
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = URL_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = HASHTAG_RE.sub(" ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]


def lemmatize(tokens: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    text = clean_text(text)
    tokens = tokenize(text)
    lemmas = lemmatize(tokens)
    return " ".join(lemmas)
