"""День 4 — Улучшение baseline"""
import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from classification_day02 import LABEL_COL, TEXT_COL, preprocess_dataframe

DATA_PATH = "data/dataset.csv"
RANDOM_STATE = 42

def tfidf(**kwargs):
    """TF-IDF с параметрами baseline по умолчанию"""
    params = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    params.update(kwargs)
    return TfidfVectorizer(**params)

def lr(**kwargs):
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **kwargs)

def make_experiments():
    """Список экспериментов"""
    def pipe(vec, clf):
        return Pipeline([("vec", vec), ("clf", clf)])

    return [
        ("baseline", "TF-IDF (1,2) + LR", pipe(tfidf(), lr())),
        # другие n-граммы 
        ("n-граммы", "TF-IDF (1,1) + LR", pipe(tfidf(ngram_range=(1, 1)), lr())),
        ("n-граммы", "TF-IDF (1,3) + LR", pipe(tfidf(ngram_range=(1, 3)), lr())),
        # другая векторизация 
        ("векторизация", "TF-IDF + stop-слова + LR", pipe(tfidf(stop_words="english"), lr())),
        ("векторизация", "CountVectorizer (1,2) + LR",
            pipe(CountVectorizer(ngram_range=(1, 2), min_df=2), lr())),
        ("векторизация", "TF-IDF символьные 2-5 + LR",
            pipe(tfidf(analyzer="char_wb", ngram_range=(2, 5)), lr())),
        ("векторизация", "TF-IDF слова + символы + LR",
            pipe(FeatureUnion([("word", tfidf()), ("char", tfidf(analyzer="char_wb", ngram_range=(2, 5)))]), lr())),
        # другая модель 
        ("модель", "TF-IDF + LinearSVC", pipe(tfidf(), LinearSVC(random_state=RANDOM_STATE))),
        ("модель", "TF-IDF + ComplementNB", pipe(tfidf(), ComplementNB())),
        # гиперпараметры 
        ("гиперпараметры", "TF-IDF + LR (C=10)", pipe(tfidf(), lr(C=10))),
        ("гиперпараметры", "TF-IDF + LR (C=100)", pipe(tfidf(), lr(C=100))),
    ]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    df = pd.read_csv(path)
    clean_df, label2id, id2label, _ = preprocess_dataframe(df)
    names = [id2label[i] for i in sorted(id2label)]

    X_train, X_test, y_train, y_test = train_test_split(
        clean_df[TEXT_COL], clean_df[LABEL_COL], test_size=0.2,
        stratify=clean_df[LABEL_COL], random_state=RANDOM_STATE,
    )
    print(f"train: {len(X_train)}, test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    print("\n=== Эксперименты (macro F1) ===")
    rows = []
    fitted = {}
    for group, name, model in make_experiments():
        start = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        test_f1 = f1_score(y_test, model.predict(X_test), average="macro")
        rows.append({
            "group": group, "name": name,
            "cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std(),
            "test_f1": test_f1, "seconds": time.time() - start,
        })
        fitted[name] = model
        print(f"{name:32s} CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | test: {test_f1:.4f} | {rows[-1]['seconds']:.1f} с")

    # Подбор гиперпараметров по сетке 
    print("\nПодбор гиперпараметров (GridSearchCV, LinearSVC) ")
    grid = GridSearchCV(
        Pipeline([("vec", tfidf()), ("clf", LinearSVC(random_state=RANDOM_STATE))]),
        param_grid={
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2],
            "clf__C": [0.3, 1, 3],
        },
        cv=cv, scoring="f1_macro", n_jobs=1,
    )
    grid.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"CV macro F1: {grid.best_score_:.4f}")
    grid_name = "TF-IDF + LinearSVC (GridSearch)"
    rows.append({
        "group": "гиперпараметры", "name": grid_name,
        "cv_f1_mean": grid.best_score_, "cv_f1_std": float("nan"),
        "test_f1": f1_score(y_test, grid.predict(X_test), average="macro"), "seconds": float("nan"),
    })
    fitted[grid_name] = grid.best_estimator_

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    results.to_csv("results/day04_experiments.csv", index=False)

    best = results.loc[results["cv_f1_mean"].idxmax()]
    base = results[results["group"] == "baseline"].iloc[0]
    best_model = fitted[best["name"]]
    y_pred = best_model.predict(X_test)

    print("\n=== Итог ===")
    print(f"Baseline:  CV {base.cv_f1_mean:.4f} | test macro F1 {base.test_f1:.4f}")
    print(f"Лучший:    {best['name']} | CV {best.cv_f1_mean:.4f} | test macro F1 {best.test_f1:.4f}")
    print(f"Прирост на test: {best.test_f1 - base.test_f1:+.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=names, digits=4)}")

    metrics = {
        "model": best["name"],
        "cv_macro_f1": round(float(best.cv_f1_mean), 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "baseline_test_macro_f1": round(float(base.test_f1), 4),
    }
    with open("results/best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    joblib.dump(best_model, "models/best_model.joblib")

    with open("results/day04_results.txt", "w", encoding="utf-8") as f:
        f.write("=== ЭКСПЕРИМЕНТЫ ===\n\n")
        f.write(results.round(4).to_string(index=False))
        f.write(f"\n\nЛучший по CV: {best['name']}\n")
        f.write(f"Baseline test macro F1: {base.test_f1:.4f}\nЛучший test macro F1: {best.test_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=names, digits=4))
    print("Сохранено: results/day04_experiments.csv, day04_results.txt, best_metrics.json, models/best_model.joblib")


if __name__ == "__main__":
    main()

