import math
from typing import List
import pandas as pd
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

db = MorphologyDB.builtin_db()
analyzer = Analyzer(db)

BASIC_ARABIC_STOPWORDS = set("""
في على عن من إلى الى و أو او ثم كما قد لقد لما لن لم لا ما ألا ال هذا هذه ذلك تلك هو هي هم هن نحن انت أنت أنتم أنتن كان كانت يكون تكون يكونون كنتم كانو حيث عندما اذا إن أن
""".split())

def _tokenize(text: str) -> List[str]:

    text = "" if text is None else str(text)
    tokens = text.split()
    return [t.strip() for t in tokens if t.strip()]

# Feature 21: Brunet's W
def brunet_w(text: str) -> float:

    tokens = _tokenize(text)
    N = len(tokens)
    if N <= 1:
        return 0.0

    V = len(set(tokens))
    logN = math.log(N)
    if logN <= 0:
        return 0.0

    return V / (logN ** 0.172)

# Feature 44: Proper Nouns Count
def count_proper_nouns(text: str) -> int:
    if not text:
        return 0

    tokens = str(text).split()
    proper_count = 0

    for tok in tokens:
        analyses = analyzer.analyze(tok)
        for a in analyses:
            if a.get("pos") == "noun_prop":
                proper_count += 1
                break

    return proper_count

# Feature 67: Singular Words Count
def count_singular_words(text: str) -> int:
    if not text:
        return 0

    tokens = str(text).split()
    singular_count = 0

    for tok in tokens:
        analyses = analyzer.analyze(tok)
        for a in analyses:
            pos = a.get("pos")
            num = a.get("num")
            if pos in {"noun", "noun_prop"} and num == "s":
                singular_count += 1
                break

    return singular_count


# Feature 90: Entity Diversity
def entity_diversity(text: str) -> float:
    if not text:
        return 0.0

    tokens = str(text).split()
    proper_entities = []

    for tok in tokens:
        analyses = analyzer.analyze(tok)
        for a in analyses:
            if a.get("pos") == "noun_prop":
                proper_entities.append(tok)
                break

    total = len(proper_entities)
    if total == 0:
        return 0.0

    unique = len(set(proper_entities))
    return unique / total


def add_stylometric_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    df = df.copy()
    df["brunet_w"] = df[text_col].apply(brunet_w)
    df["proper_nouns_count"] = df[text_col].apply(count_proper_nouns)
    df["singular_words_count"] = df[text_col].apply(count_singular_words)
    df["entity_diversity"] = df[text_col].apply(entity_diversity)
    return df