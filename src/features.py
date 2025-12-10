"""
features.py

ONLY the 4 required features for the MSIS822 project:

    - Feature 21: Brunet's W
    - Feature 44: Proper Nouns Count (heuristic)
    - Feature 67: Singular Words Count (heuristic)
    - Feature 90: Entity Diversity (based on proper nouns)

All implemented with simple heuristics, no external heavy NLP models.
"""

import math
from typing import List
import pandas as pd

BASIC_ARABIC_STOPWORDS = set("""
في على عن من إلى الى و أو او ثم كما قد لقد لما لن لم لا ما ألا ال هذا هذه ذلك تلك هو هي هم هن نحن انت أنت أنتم أنتن كان كانت يكون تكون يكونون كنتم كانو حيث عندما اذا إن أن
""".split())


# =========================
# Helpers
# =========================

def _tokenize(text: str) -> List[str]:
    """Simple whitespace-based tokenization."""
    text = "" if text is None else str(text)
    tokens = text.split()
    return [t.strip() for t in tokens if t.strip()]


# =========================
# Feature 21: Brunet's W
# =========================

def brunet_w(text: str) -> float:
    """
    Brunet's W measure of vocabulary richness.

    W = V / (ln(N))^0.172
        N = number of tokens
        V = number of unique tokens

    For very short texts (N <= 1) or edge cases, return 0.0 (to avoid division by zero).
    """
    tokens = _tokenize(text)
    N = len(tokens)
    if N <= 1:
        return 0.0

    V = len(set(tokens))
    logN = math.log(N)
    if logN <= 0:
        return 0.0

    return V / (logN ** 0.172)


# =========================
# Feature 44: Proper Nouns Count (heuristic)
# =========================

def count_proper_nouns(text: str) -> int:
    """
    Heuristic approximation for Proper Nouns Count (Feature 44).

    Idea:
        - نستخدم النص المنظَّف (clean_text) لكن الهيوريستك تقريبية:
            * الكلمة ليست stopword
            * طولها >= 4
            * لا تبدو كجمع شائع (ون / ين / ات)
        - نعتبر هذه الكلمات كـ "اسم علم" تقريبياً.

    مهم:
        هذه ليست دقيقة لغوياً، لكن تعطي نمطاً ثابتاً يمكن استخدامه كـ feature.
    """
    tokens = _tokenize(text)
    proper_like = []
    for tok in tokens:
        if tok in BASIC_ARABIC_STOPWORDS:
            continue
        if len(tok) < 4:
            continue
        if tok.endswith("ون") or tok.endswith("ين") or tok.endswith("ات"):
            continue
        proper_like.append(tok)
    return len(proper_like)


# =========================
# Feature 67: Singular Words Count (heuristic)
# =========================

def count_singular_words(text: str) -> int:
    """
    Heuristic approximation for Singular Words Count (Feature 67).

    Idea:
        - نعتبر الكلمات التي تنتهي بـ (ون / ين / ات) كجمع.
        - الباقي نعتبره مفرد (تقريبية).
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0

    plural_count = 0
    for tok in tokens:
        if len(tok) <= 3:
            continue
        if tok.endswith("ون") or tok.endswith("ين") or tok.endswith("ات"):
            plural_count += 1

    singular_count = len(tokens) - plural_count
    return singular_count


# =========================
# Feature 90: Entity Diversity (heuristic)
# =========================

def entity_diversity(text: str) -> float:
    """
    Heuristic approximation for Entity Diversity (Feature 90).

    Idea:
        - نستخدم نفس تعريف "proper-like tokens" من count_proper_nouns.
        - entities = نفس القائمة التي حسبناها كـ proper nouns.
        - diversity = عدد الكيانات الفريدة / عدد الكيانات الكلي
    """
    tokens = _tokenize(text)

    # نعيد استخدام منطق proper nouns (لكن بدون تكرار الكود)
    proper_like = []
    for tok in tokens:
        if tok in BASIC_ARABIC_STOPWORDS:
            continue
        if len(tok) < 4:
            continue
        if tok.endswith("ون") or tok.endswith("ين") or tok.endswith("ات"):
            continue
        proper_like.append(tok)

    total = len(proper_like)
    if total == 0:
        return 0.0

    unique = len(set(proper_like))
    return unique / total


# =========================
# Apply to DataFrame
# =========================

def add_stylometric_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add ONLY the 4 required stylometric features + (leave label as is).

    Columns added:
        - brunet_w
        - proper_nouns_count
        - singular_words_count
        - entity_diversity
    """
    df = df.copy()
    df["brunet_w"] = df[text_col].apply(brunet_w)
    df["proper_nouns_count"] = df[text_col].apply(count_proper_nouns)
    df["singular_words_count"] = df[text_col].apply(count_singular_words)
    df["entity_diversity"] = df[text_col].apply(entity_diversity)
    return df