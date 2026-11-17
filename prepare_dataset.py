"""Скачивает новости AG News (открытый датасет, 4 класса) и сохраняет выборку в data/dataset.csv"""
import io
import os
import sys
import urllib.request

import pandas as pd

URL = "https://huggingface.co/api/datasets/fancyzhx/ag_news/parquet/default/train/0.parquet"
OUT_PATH = "data/dataset.csv"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def main(per_class=500):
    print("Скачиваю AG News...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        df = pd.read_parquet(io.BytesIO(resp.read()))

    # Одинаковое число новостей каждого класса; random_state — одна и та же выборка при каждом запуске.
    sample = (
        df.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=42)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    sample["label_name"] = sample["label"].map(LABEL_NAMES)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)
    print(f"Сохранено {len(sample)} строк в {OUT_PATH}")
    print(sample["label_name"].value_counts().to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
