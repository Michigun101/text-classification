"""День 3 — Baseline модель классификации"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from days.classification_day02 import LABEL_COL, TEXT_COL, preprocess_dataframe

DATA_PATH = "data/dataset.csv"
METRICS_PATH = "results/baseline_metrics.json"
REPORT_PATH = "results/baseline_results.txt"
MODEL_PATH = "models/baseline_tfidf_lr.joblib"
RANDOM_STATE = 42

def build_baseline():
    """Baseline pipeline"""
    return Pipeline([
        ('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True)),
        ('clf',LogisticRegression(max_iter=1000,random_state=RANDOM_STATE))
    ])

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    print("1. Данные (preprocessing из Дня 2) ")
    df = pd.read_csv(path)
    clean_df, label2id, id2label, report = preprocess_dataframe(df)
    print(f"Строк после очистки: {len(clean_df)}")


    print("\n2. Разбиение train/test ")
    X = clean_df[TEXT_COL]
    y = clean_df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print(f"Классы в test:\n{y_test.value_counts().sort_index().to_string()}")

    print("\n3. Обучение baseline ")
    model = build_baseline()
    model.fit(X_train, y_train)
    print(f"Размер словаря TF-IDF: {len(model.named_steps['tfidf'].vocabulary_)}")

    print("\n4. Метрики на test ")
    y_pred = model.predict(X_test)
    names = [id2label[i] for i in sorted(id2label)]

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cls_report = classification_report(y_test, y_pred, target_names=names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"\n{cls_report}")
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=names, columns=names).to_string())

    print("\n5. Сохранение baseline ")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    metrics = {
        "model": "TF-IDF (1-2 grams) + LogisticRegression",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "per_class_f1": {
            name: round(f, 4)
            for name, f in zip(names, f1_score(y_test, y_pred, average=None))
        },
        "random_state": RANDOM_STATE,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== BASELINE: TF-IDF + LogisticRegression ===\n\n")
        f.write(f"train: {len(X_train)}, test: {len(X_test)}\n")
        f.write(f"Accuracy:    {accuracy:.4f}\n")
        f.write(f"Macro F1:    {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n\n")
        f.write(cls_report)
        f.write("\nConfusion matrix :\n")
        f.write(pd.DataFrame(cm, index=names, columns=names).to_string())
        f.write("\n")

    joblib.dump(model, MODEL_PATH)
    print(f"Сохранено: {METRICS_PATH}, {REPORT_PATH}, {MODEL_PATH}")

    print(f"\nBASELINE (точка отсчёта): macro F1 = {f1_macro:.4f}, accuracy = {accuracy:.4f}")



if __name__ == "__main__":
    main()