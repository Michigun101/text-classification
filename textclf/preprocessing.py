"""Очистка текста и подготовка меток (preprocessing)"""
import html
import re
import unicodedata

import pandas as pd

TEXT_COL = "text"
LABEL_COL = "label"
LABEL_NAME_COL = "label_name"

NUM_ENTITY_RE = re.compile(r"#(\d+);")                
TAG_RE = re.compile(r"<[^>]+>")                       
URL_RE = re.compile(r"https?://\S+|www\.\S+")
BACKSLASH_DOLLAR_RE = re.compile(r"\\+(?=\$)")        
BACKSLASH_RE = re.compile(r"\\+")
AGENCY_RE = re.compile(r"\s*\((?:Reuters|AP|AFP|CP)\)\s*(?:-+\s*)?", re.IGNORECASE)
AGENCY_PREFIX_RE = re.compile(r"\b(?:AP|Reuters|AFP)\s+-\s+", re.IGNORECASE)
SPACES_RE = re.compile(r"\s+")


def _decode_numeric_entity(match):
    """'#39;' -> "'". Коды 128-159 — это символы Windows-1252"""
    code = int(match.group(1))
    if 128 <= code <= 159:
        try:
            return bytes([code]).decode("cp1252")
        except UnicodeDecodeError:
            return " "
    return chr(code) if code < 0x110000 else " "


def clean_text(text, lowercase=True, remove_agency=True):
    """Очищает и нормализует один текст"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)

    text = unicodedata.normalize("NFKC", text)

    text = NUM_ENTITY_RE.sub(_decode_numeric_entity, text)
    text = html.unescape(text)

    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)

    text = BACKSLASH_DOLLAR_RE.sub("", text)
    text = BACKSLASH_RE.sub(" ", text)

    if remove_agency:
        text = AGENCY_RE.sub(" ", text)
        text = AGENCY_PREFIX_RE.sub(" ", text)

    if lowercase:
        text = text.lower()

    return SPACES_RE.sub(" ", text).strip()


def build_label_mapping(df, label_col=LABEL_COL, name_col=LABEL_NAME_COL):
    """Строит словари label2id (название -> число) и id2label (число -> название)"""
    pairs = df[[label_col, name_col]].drop_duplicates().sort_values(label_col)
    if pairs[label_col].duplicated().any() or pairs[name_col].duplicated().any():
        raise ValueError(f"Метки неоднозначны:\n{pairs}")
    ids = pairs[label_col].astype(int).tolist()
    if ids != list(range(len(ids))):
        raise ValueError(f"Номера классов должны идти подряд с 0, а не {ids}")

    id2label = {int(i): name for i, name in zip(pairs[label_col], pairs[name_col])}
    label2id = {name: i for i, name in id2label.items()}
    return label2id, id2label


def preprocess_dataframe(df, text_col=TEXT_COL, label_col=LABEL_COL, name_col=LABEL_NAME_COL,
                         min_words=3, **clean_kwargs):
    """Полный preprocessing датасета"""
    report = {"rows_before": len(df)}

    df = df.dropna(subset=[text_col, label_col]).copy()
    report["dropped_nan"] = report["rows_before"] - len(df)

    label2id, id2label = build_label_mapping(df, label_col, name_col)
    df[label_col] = df[label_col].astype(int)

    df[text_col] = df[text_col].map(lambda t: clean_text(t, **clean_kwargs))

    too_short = df[text_col].str.split().str.len().fillna(0) < min_words
    report["dropped_short"] = int(too_short.sum())
    df = df[~too_short]

    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    report["dropped_duplicates"] = before - len(df)

    df = df[[text_col, label_col, name_col]].reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, label2id, id2label, report
