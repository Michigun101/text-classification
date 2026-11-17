"""День 1 — Загрузка датасета и первичный анализ (EDA)"""
import sys

import pandas as pd

DATA_PATH = "data/dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_NAME_COL = "label_name"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    print("1. Загрузка датасета ")
    df = pd.read_csv(path)
    print(f"Файл: {path}")
    print(df.head())

    print("\n2. Размер, типы колонок, пропуски ")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
    print("\nТипы колонок:")
    print(df.dtypes.to_string())
    print("\nПропуски по колонкам:")
    print(df.isna().sum().to_string())
    # Дубликаты текстов
    print(f"\nДубликатов текстов: {df[TEXT_COL].duplicated().sum()}")
    # Пустые или состоящие из пробелов тексты 
    print(f"Пустых текстов: {(df[TEXT_COL].astype(str).str.strip() == '').sum()}")

    print("\n3. Распределение классов ")
    counts = df[LABEL_COL].value_counts().sort_index()
    shares = df[LABEL_COL].value_counts(normalize=True).sort_index() * 100
    names = df.drop_duplicates(LABEL_COL).set_index(LABEL_COL)[LABEL_NAME_COL]
    for label in counts.index:
        print(f"{label} ({names[label]}): {counts[label]} ({shares[label]:.1f}%)")
    print(f"Число классов: {df[LABEL_COL].nunique()}")
    print(f"Дисбаланс (max/min): {counts.max() / counts.min():.2f}")

    print("\n4. Длины текстов ")
    df["n_chars"] = df[TEXT_COL].astype(str).str.len()
    df["n_words"] = df[TEXT_COL].astype(str).str.split().str.len()
    print(df[["n_chars", "n_words"]].describe().round(1).to_string())
    print("\nСредние длины по классам (слов):")
    print(df.groupby(LABEL_NAME_COL)["n_words"].agg(["mean", "median", "min", "max"]).round(1).to_string())
    print(f"\nТекстов длиннее 100 слов: {(df['n_words'] > 100).sum()}")

    print("\n5. Примеры текстов из разных классов ")
    for label in counts.index:
        print(f"\n  Класс {label} ({names[label]}) ")
        samples = df[df[LABEL_COL] == label].sample(2, random_state=42)
        for text in samples[TEXT_COL]:
            print(f"* {text[:200]}")

    print("\n6. Наблюдения о данных ")
    texts = df[TEXT_COL].astype(str)
    print(f"С HTML-сущностями (#NN; или &xx;): {texts.str.contains(r'#\d+;|&[a-z]+;').sum()}")
    print(f"С пометкой агентства (Reuters)/(AP): {texts.str.contains(r'\((?:Reuters|AP)\)').sum()}")
    print(f"С символом \\ : {texts.str.contains(chr(92), regex=False).sum()}")
    print("Доля текстов с пометкой агентства по классам:")
    print(df.assign(agency=texts.str.contains(r"\((?:Reuters|AP)\)")).groupby(LABEL_NAME_COL)["agency"].mean().round(3).to_string())


if __name__ == "__main__":
    main()

