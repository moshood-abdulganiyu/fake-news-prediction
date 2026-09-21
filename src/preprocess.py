"""Text cleaning for the fake news classifier.

Feature decision (see notebooks/01_eda.ipynb for the numbers behind it):
- Subject is excluded entirely. It's a 100% leak in this dataset (every
  subject tag maps to exactly one label), not a real signal.
- Reuters dateline prefix is stripped from `text` before cleaning, since
  its presence alone predicts "real" with 100% precision in this dataset
  and would let the model shortcut on formatting instead of content.
- Date is not used as a feature.
- Primary content field is title + text (post dateline-strip). A
  title-only variant is trained separately in Step 4 for comparison, to
  show the text body earns its place rather than assuming it does.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)

_REUTERS_PATTERN = re.compile(r"^\s*\S.*?\(Reuters\)\s*-\s*")
_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z]")
_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def strip_reuters_dateline(text: str) -> str:
    """Remove a leading Reuters dateline like 'WASHINGTON (Reuters) - '.

    Confirmed in Step 2 EDA: 100% of rows carrying this pattern are
    labeled real in this dataset. Left in, a model learns to detect the
    string "(Reuters)" instead of anything about the article itself.
    """
    return _REUTERS_PATTERN.sub("", text, count=1)


def clean_text(text: str) -> str:
    """Lowercase, strip non-alphabetic characters, remove stopwords, stem.

    Same approach as the old notebook's stemming() function. Applied
    identically to every field that feeds the model so results stay
    comparable across the title-only and title+text variants.
    """
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = text.lower()
    words = text.split()
    words = [_STEMMER.stem(w) for w in words if w not in _STOPWORDS]
    return " ".join(words)


def build_content(title: str, text: str, include_text: bool = True) -> str:
    """Combine and clean the fields that go into the model.

    Args:
        title: Article title.
        text: Article body. Ignored if include_text is False.
        include_text: When False, produces the title-only variant used
            for comparison in Step 4.

    Returns:
        Cleaned, stemmed content string ready for TF-IDF vectorization.
    """
    title = title or ""
    text = text or ""

    if include_text:
        text = strip_reuters_dateline(text)
        raw = f"{title} {text}"
    else:
        raw = title

    return clean_text(raw)
