"""День 5 — Error analysis и интерпретация результатов"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from days.classification_day02 import LABEL_COL, TEXT_COL, preprocess_dataframe
from days.classification_day04 import RANDOM_STATE, lr, tfidf

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/best_model.joblib"

def load_best_model(X_train, y_train):
    """Загружает лучшую модель Дня 4"""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    print("models/best_model.joblib не найден")
    model = Pipeline([("vec", tfidf(stop_words="english")), ("clf", lr())])
    return model.fit(X_train, y_train)


def top_terms_per_class(model, names, k=10):
    """Слова с наибольшим положительным весом для каждого класса"""
    terms = np.array(model.named_steps['vec'].get_feature_names_out())
    coef = model.named_steps['clf'].coef_
    return {name: terms[np.argsort(coef[i])[::-1][:k]].tolist() for i, name in enumerate(names)}

def explain_prediction(model, text, class_index, k=5):
    """Слова текста, с наибольшим вкладом в класс class_index """
    vec, clf = model.named_steps['vec'], model.named_steps["clf"]
    row = vec.transform([text])
    terms = np.array(vec.get_feature_names_out())
    contrib = row.multiply(clf.coef_[class_index]).toarray()[0]
    top = np.argsort(contrib)[::-1][:k]
    return [(str(terms[i]), round(float(contrib[i]), 2)) for i in top if contrib[i] > 0]


def main():
    df = pd.read_csv(DATA_PATH)
    clean_df, label2id, id2label, _ = preprocess_dataframe(df)
    names = [id2label[i] for i in sorted(id2label)]

    X_train, X_test, y_train, y_test = train_test_split(
        clean_df[TEXT_COL], clean_df[LABEL_COL], test_size=0.2,
        stratify=clean_df[LABEL_COL], random_state=RANDOM_STATE,
    )
    model = load_best_model(X_train, y_train)

    out = []
    def log(line=""):
        print(line)
        out.append(line)

    # Задача 1: найти ошибки 
    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    res = pd.DataFrame({
        "text": X_test.values,
        "true": y_test.values,
        "pred": pred,
        "confidence": proba.max(axis=1),
    })
    res["true_name"] = res["true"].map(id2label)
    res["pred_name"] = res["pred"].map(id2label)
    res["n_words"] = res["text"].str.split().str.len()
    errors = res[res["true"] != res["pred"]].copy()

    log(" 1. Ошибки ")
    log(f"Всего test: {len(res)}, ошибок: {len(errors)} ({len(errors) / len(res) * 100:.1f}%), "
        f"accuracy: {1 - len(errors) / len(res):.4f}")
    log("\nОшибки по истинному классу:")
    for name, n in errors["true_name"].value_counts().items():
        log(f"  {name:10s} {n}")

    # Задача 2: confusion matrix 
    log("\n2. Confusion matrix ")
    cm = confusion_matrix(res["true"], res["pred"])
    log(pd.DataFrame(cm, index=names, columns=names).to_string())

    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title("Confusion matrix (число текстов)")
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Нормированная по истинному классу")
    for ax in axes:
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истина")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=120)
    plt.close()
    log("Сохранено: results/confusion_matrix.png")

    # Самые частые пары путаницы
    pairs = errors.groupby(["true_name", "pred_name"]).size().sort_values(ascending=False)
    log("\nСамые частые ошибки (истина -> предсказано):")
    for (t, p), n in pairs.head(6).items():
        log(f"  {t} -> {p}: {n}")

    # Задача 3: примеры ошибок 
    log("\n3. Примеры ошибок")
    for (t, p), n in pairs.head(3).items():
        log(f"\n--- {t} -> {p} ({n} ошибок) ---")
        subset = errors[(errors["true_name"] == t) & (errors["pred_name"] == p)]
        for _, row in subset.sample(min(4, len(subset)), random_state=RANDOM_STATE).iterrows():
            log(f"[уверенность {row['confidence']:.2f}] {row['text'][:220]}")
            words = explain_prediction(model, row["text"], row["pred"])
            log(f"    слова за «{p}»: {words}")

    # Задача 4: failure cases 
    log("\n4. Failure cases ")
    ok = res[res["true"] == res["pred"]]
    log(f"Средняя уверенность: верные {ok['confidence'].mean():.3f}, ошибочные {errors['confidence'].mean():.3f}")
    log(f"Средняя длина (слов): верные {ok['n_words'].mean():.1f}, ошибочные {errors['n_words'].mean():.1f}")

    bins = pd.cut(res["confidence"], [0, 0.5, 0.7, 0.9, 1.0])
    by_conf = res.groupby(bins, observed=True).apply(
        lambda g: pd.Series({"n": len(g), "accuracy": (g["true"] == g["pred"]).mean()}),
        include_groups=False,
    )
    log("\nТочность по уверенности модели:")
    log(by_conf.round(3).to_string())

    short = res["n_words"] <= res["n_words"].quantile(0.25)
    log(f"\nТочность на самых коротких 25% текстов (<= {res['n_words'].quantile(0.25):.0f} слов): "
        f"{(res[short]['true'] == res[short]['pred']).mean():.3f}; на остальных: "
        f"{(res[~short]['true'] == res[~short]['pred']).mean():.3f}")

    log("\nТоп-слова каждого класса:")
    for name, terms in top_terms_per_class(model, names).items():
        log(f"  {name:10s} {', '.join(terms)}")

    errors.drop(columns=["true", "pred"]).to_csv("results/errors.csv", index=False)
    with open("results/error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\nСохранено: results/errors.csv, results/error_analysis.txt")


if __name__ == "__main__":
    main()

