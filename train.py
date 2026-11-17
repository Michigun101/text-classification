"""Обучение: данные -> preprocessing -> train/test -> обучение -> метрики -> сохранение модели"""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from textclf import TextClassifier, build_pipeline, preprocess_dataframe
from textclf.model import CLEAN_KWARGS, RANDOM_STATE
from textclf.preprocessing import LABEL_COL, TEXT_COL


def main():
    parser = argparse.ArgumentParser(description="Обучение классификатора новостей")
    parser.add_argument("--data", default="data/dataset.csv", help="CSV с колонками text, label, label_name")
    parser.add_argument("--model", default="models/text_classifier.joblib", help="куда сохранить модель")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    clean_df, label2id, id2label, report = preprocess_dataframe(df, **CLEAN_KWARGS)
    print(f"Данные: {report['rows_before']} -> {report['rows_after']} строк после очистки "
          f"(дубликатов удалено: {report['dropped_duplicates']})")

    X_train, X_test, y_train, y_test = train_test_split(
        clean_df[TEXT_COL], clean_df[LABEL_COL], test_size=0.2,
        stratify=clean_df[LABEL_COL], random_state=RANDOM_STATE,
    )
    print(f"train: {len(X_train)}, test: {len(X_test)}")

    pipeline = build_pipeline().fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    names = [id2label[i] for i in sorted(id2label)]
    metrics = {
        "model": "TF-IDF (1-2 grams, stop-words) + LogisticRegression",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "per_class_f1": {n: round(f, 4) for n, f in zip(names, f1_score(y_test, y_pred, average=None))},
    }
    cls_report = classification_report(y_test, y_pred, target_names=names, digits=4)
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=names, columns=names)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}, macro F1: {metrics['macro_f1']:.4f}\n")
    print(cls_report)
    print("Confusion matrix (строки — истина, столбцы — предсказание):")
    print(cm.to_string())

    os.makedirs("results", exist_ok=True)
    with open("results/final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open("results/final_report.txt", "w", encoding="utf-8") as f:
        f.write(cls_report + "\nConfusion matrix (строки — истина, столбцы — предсказание):\n" + cm.to_string() + "\n")

    TextClassifier.from_pipeline(pipeline, id2label, metadata=metrics).save(args.model)
    print(f"\nМодель сохранена: {args.model}")


if __name__ == "__main__":
    main()
