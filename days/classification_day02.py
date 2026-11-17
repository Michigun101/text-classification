"""День 2 — Очистка текста и подготовка признаков"""
import html
import json
import re
import sys
import unicodedata

import pandas as pd

DATA_PATH = "data/dataset.csv"
CLEAN_PATH = "data/dataset_clean.csv"
MAPPING_PATH = "data/label_mapping.json"
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_NAME_COL = "label_name"

NUM_ENTITY_RE = re.compile(r"#(\d+);")                # '#39;' 
TAG_RE = re.compile(r"<[^>]+>")                       # HTML-теги
URL_RE = re.compile(r"https?://\S+|www\.\S+")
BACKSLASH_DOLLAR_RE = re.compile(r"\\+(?=\$)")        # '\$10' -> '$10'
BACKSLASH_RE = re.compile(r"\\+")

AGENCY_RE = re.compile(r"\s*\((?:Reuters|AP|AFP|CP)\)\s*(?:-+\s*)?", re.IGNORECASE)
AGENCY_PREFIX_RE = re.compile(r"\b(?:AP|Reuters|AFP)\s+-\s+", re.IGNORECASE)
SPACES_RE = re.compile(r"\s+")

def _decode_numeric_entity(match):
    """Декодирование символов Windows-1252"""
    code = int(match.group(1))
    if 128 <= code <= 159:
        try:
            return bytes([code]).decode('cp1252')
        except UnicodeDecodeError:
            return " "
    return chr(code) if code < 0x110000 else " "

def clean_text(text, lowercase=True, remove_agency=True):
    """Очищает и нормализует один текст"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    text = str(text)
    # Приводим разные написания одного символа к одному виду
    text = unicodedata.normalize("NFKC", text)

    text = NUM_ENTITY_RE.sub(_decode_numeric_entity,text)
    text = html.unescape(text)

    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)

    text = BACKSLASH_DOLLAR_RE.sub("", text)
    text = BACKSLASH_RE.sub(" ", text)

    # Подписи агентств 
    if remove_agency:
        text = AGENCY_RE.sub(" ", text)
        text = AGENCY_PREFIX_RE.sub(" ", text)

    if lowercase:
        text = text.lower()

    return SPACES_RE.sub(" ", text).strip()

def build_label_mapping(df, label_col=LABEL_COL, name_col=LABEL_NAME_COL):
    """Строит словари label2id (название -> число) и id2label (число -> название)"""
    pairs = df[[label_col,name_col]].drop_duplicates().sort_values(label_col)
    ids = pairs[label_col].astype(int).tolist()

    id2label = {int(i): name for i, name in zip(pairs[label_col], pairs[name_col])}
    label2id = {name: i for i, name in id2label.items()}
    return label2id, id2label

def preprocess_dataframe(df, text_col=TEXT_COL, label_col=LABEL_COL, name_col=LABEL_NAME_COL,
                         min_words=3, **clean_kwargs):
    """Полный preprocessing датасета."""
    report = {"rows_before": len(df)}
    # Строки без метки или без текста 
    df = df.dropna(subset=[text_col, label_col]).copy()
    report["dropped_nan"] = report["rows_before"] - len(df)

    label2id, id2label = build_label_mapping(df, label_col, name_col)
    df[label_col] = df[label_col].astype(int)

    df[text_col] = df[text_col].map(lambda t: clean_text(t, **clean_kwargs))

    # после очистки остался пустой или слишком короткий текст
    too_short = df[text_col].str.split().str.len().fillna(0) < min_words
    report["dropped_short"] = int(too_short.sum())
    df = df[~too_short]

    # Дубликаты текстов
    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    report["dropped_duplicates"] = before - len(df)

    df = df[[text_col, label_col, name_col]].reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, label2id, id2label, report

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    print("\nPreprocessing датасета ")
    df = pd.read_csv(path)
    clean_df, label2id, id2label, report = preprocess_dataframe(df)
    print(f"Отчёт: {report}")
    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")

    print("\nПроверка результата ")
    texts = clean_df[TEXT_COL]
    print(f"Остались HTML-сущности: {texts.str.contains(r'#\d+;|&[a-z]+;').sum()}")
    print(f"Остались теги: {texts.str.contains('<|>').sum()}")
    print(f"Остались '\\': {texts.str.contains(chr(92), regex=False).sum()}")
    print(f"Остались подписи агентств: {texts.str.contains(r'\((?:reuters|ap|afp|cp)\)').sum()}")
    print(f"Средняя длина в словах: {texts.str.split().str.len().mean():.1f}")
    print(f"Классы после очистки:\n{clean_df[LABEL_COL].value_counts().sort_index().to_string()}")

    print("\nПримеры до и после (первые 3 строки):")
    for i in range(3):
        print(f"\nДО:    {df[TEXT_COL][i][:150]!r}")
        print(f"ПОСЛЕ: {clean_df[TEXT_COL][i][:150]!r}")

    clean_df.to_csv(CLEAN_PATH, index=False)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {CLEAN_PATH}, {MAPPING_PATH}")


if __name__ == "__main__":
    main()