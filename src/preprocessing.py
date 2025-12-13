do_stemming=False
import re
from typing import List, Iterable, Set, Optional
from nltk.stem.isri import ISRIStemmer


# Arabic diacritics (harakat)
ARABIC_DIACRITICS_PATTERN = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)

PUNCTUATION_PATTERN = re.compile(
    r"["
    r"\u060C"  # ،
    r"\u061B"  # ؛
    r"\u061F"  # ؟
    r"\u066A-\u066D"  # ٪٫٬٭
    r"\u06D4"  # ۔
    r"\u2013-\u2014"  # – —
    r"\u2018-\u2019"  # ‘ ’
    r"\u201C-\u201D"  # “ ”
    r"!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~"
    r"]"
)

DIGITS_PATTERN = re.compile(r"[\d\u0660-\u0669]+")

ARABIC_NORMALIZATION_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ى": "ي",
    "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
}

def remove_diacritics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ARABIC_DIACRITICS_PATTERN.sub("", text)


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = ''.join(ARABIC_NORMALIZATION_MAP.get(ch, ch) for ch in text)
    return normalized


def remove_punctuation(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return PUNCTUATION_PATTERN.sub(" ", text)


def remove_digits(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return DIGITS_PATTERN.sub(" ", text)


def keep_arabic_only(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\u0600-\u06FF\s]", " ", text)


def basic_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t for t in text.split() if t.strip()]


def remove_stopwords(tokens: Iterable[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    if stopwords is None:
        return list(tokens)
    return [t for t in tokens if t not in stopwords]


try:
    _ISRI_STEMMER = ISRIStemmer()
except Exception:
    _ISRI_STEMMER = None


def stem_tokens(tokens: Iterable[str], use_isri: bool = True) -> List[str]:
    tokens = list(tokens)
    if not use_isri or _ISRI_STEMMER is None:
        return tokens
    return [_ISRI_STEMMER.stem(t) for t in tokens]

def preprocess_text(
        text: str,
        stopwords: Optional[Set[str]] = None,
        *,
        normalize: bool = True,
        remove_harakat: bool = True,
        remove_punct: bool = True,
        remove_nums: bool = True,
        arabic_only: bool = False,
        do_stemming: bool = False,
        return_tokens: bool = False,
):

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    if normalize:
        text = normalize_arabic(text)
    if remove_harakat:
        text = remove_diacritics(text)
    if remove_punct:
        text = remove_punctuation(text)
    if remove_nums:
        text = remove_digits(text)
    if arabic_only:
        text = keep_arabic_only(text)

    tokens = basic_tokenize(text)
    tokens = remove_stopwords(tokens, stopwords)

    if do_stemming:
        tokens = stem_tokens(tokens)

    return tokens if return_tokens else " ".join(tokens)