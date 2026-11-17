"""День 6 — Inference pipeline и сохранение модели"""
import json
import os
import sys
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification_day02 import LABEL_COL, TEXT_COL, clean_text, preprocess_dataframe
from classification_day04 import RANDOM_STATE, lr, tfidf

DATA_PATH = "data/dataset.csv"
BEST_MODEL_PATH = "models/best_model.joblib"       
MODEL_PATH = "models/text_classifier.joblib"       
CLEAN_KWARGS = {"lowercase": True, "remove_agency": True}

class TextClassifier:
    def __init__(self, vectorizer, classifier, id2label, clean_kwargs, metadata=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.clean_kwargs = clean_kwargs
        self.metadata = metadata or {}

    def save(self, path=MODEL_PATH):
        """Сохраняет vectorizer, классификатор, названия классов и параметры preprocessing в один файл"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "id2label": self.id2label,
            "clean_kwargs": self.clean_kwargs,
            "metadata": self.metadata,
            "sklearn_version": sklearn.__version__,
        }, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        """Загружает модель"""
        data = joblib.load(path)
        if data["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"Модель обучена в scikit-learn {data['sklearn_version']}, а установлен {sklearn.__version__}: "
                "результаты могут отличаться."
            )
        return cls(data["vectorizer"], data["classifier"], data["id2label"],
                   data["clean_kwargs"], data["metadata"])

    def predict(self, texts):
        """Классифицирует один текст или список текстов"""
        if isinstance(texts,str):
            texts = [texts]
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("predict ожидает строку или список строк")

        cleaned = [clean_text(t, **self.clean_kwargs) for t in texts]
        valid = [i for i,t in enumerate(cleaned) if t]

        results = [
            {"text": t, "label": None, "label_id": None, "confidence": None, "probabilities": None,
                "error": "empty text after preprocessing"}
            for t in texts
        ]
        if valid:
            X = self.vectorizer.transform([cleaned[i] for i in valid])
            proba = self.classifier.predict_proba(X)
            for row, i in zip(proba, valid):
                best = int(row.argmax())
                results[i] = {
                    "text": texts[i],
                    "label": self.id2label[best],
                    "label_id": best,
                    "confidence": round(float(row[best]), 4),
                    "probabilities": {self.id2label[j]: round(float(p), 4) for j, p in enumerate(row)},
                    "error": None,
                }
        return results

def build_and_save():
    df = pd.read_csv(DATA_PATH)
    clean_df, label2id, id2label, _ = preprocess_dataframe(df, **CLEAN_KWARGS)
    X_train, X_test, y_train, y_test = train_test_split(
        clean_df[TEXT_COL], clean_df[LABEL_COL], test_size=0.2,
        stratify=clean_df[LABEL_COL], random_state=RANDOM_STATE,
    )

    if os.path.exists(BEST_MODEL_PATH):
        pipeline = joblib.load(BEST_MODEL_PATH)
    else:
        print("models/best_model.joblib не найден")
        pipeline = Pipeline([("vec", tfidf(stop_words="english")), ("clf", lr())]).fit(X_train, y_train)

    metrics = {}
    if os.path.exists("results/best_metrics.json"):
        with open("results/best_metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)

    classifier = TextClassifier(
        vectorizer=pipeline.named_steps["vec"],
        classifier=pipeline.named_steps["clf"],
        id2label=id2label,
        clean_kwargs=CLEAN_KWARGS,
        metadata=metrics,
    )
    classifier.save(MODEL_PATH)
    return classifier, pipeline, X_test, y_test


def main():
    if len(sys.argv) > 1: # текст из консоли
        clf = TextClassifier.load()
        for r in clf.predict(sys.argv[1:]):
            print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    print("1. Сохранение модели ")
    clf, pipeline, X_test, y_test = build_and_save()
    print(f"Сохранено: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024:.0f} КБ)")

    print("\n2. Загрузка и проверка")
    loaded = TextClassifier.load(MODEL_PATH)
    expected = pipeline.predict(X_test)
    got = [r["label_id"] for r in loaded.predict(X_test.tolist())]
    print(f"Совпадают с исходной моделью: {list(expected) == got} ({len(got)} текстов)")
    print(f"Accuracy загруженной модели на test: {(pd.Series(got).values == y_test.values).mean():.4f}")

    print("\n3. Предсказания для новых текстов")
    new_texts = [
        "Manchester United beat Chelsea 2-1 in a thrilling Premier League match on Sunday",
        "Apple unveils new iPhone with faster chip and improved camera",
        "Oil prices jump as OPEC announces surprise production cuts, shares of energy firms rise",
        "Prime minister meets president to discuss peace talks in the Middle East",
        "Scientists discover water on a distant planet using the new space telescope",
    ]
    for r in loaded.predict(new_texts):
        print(f"[{r['label']:8s} {r['confidence']:.2f}] {r['text'][:75]}")

    print("\n4. Preprocessing на inference ")
    # Грязный текст в стиле исходных данных
    dirty = ("LONDON (Reuters) - Shares rose as the company #39;s profit beat forecasts &lt;b&gt;today&lt;/b&gt; "
             "http://example.com/news")
    cleaned = clean_text(dirty, **loaded.clean_kwargs)
    print(f"Сырой текст:   {dirty}")
    print(f"После очистки: {cleaned}")
    same = loaded.predict(dirty)[0]["probabilities"] == loaded.predict(cleaned)[0]["probabilities"]
    print(f"Предсказание для сырого текста = предсказание для очищенного: {same}")
    print(json.dumps(loaded.predict(dirty)[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

        