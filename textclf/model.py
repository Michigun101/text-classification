"""Модель и inference: сборка pipeline и класс TextClassifier."""
import os
import warnings

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from textclf.preprocessing import clean_text

RANDOM_STATE = 42
CLEAN_KWARGS = {"lowercase": True, "remove_agency": True}


def build_pipeline():
    """Лучшая конфигурация Дня 4: TF-IDF (слова и пары слов, без английских стоп-слов) + LogisticRegression."""
    return Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


class TextClassifier:
    """Готовый к использованию классификатор: сырой текст -> предсказание."""

    def __init__(self, vectorizer, classifier, id2label, clean_kwargs, metadata=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.clean_kwargs = clean_kwargs
        self.metadata = metadata or {}

    @classmethod
    def from_pipeline(cls, pipeline, id2label, clean_kwargs=CLEAN_KWARGS, metadata=None):
        """Собирает TextClassifier из обученного Pipeline (vec + clf)."""
        return cls(pipeline.named_steps["vec"], pipeline.named_steps["clf"], id2label, clean_kwargs, metadata)

    def save(self, path):
        """Сохраняет vectorizer, классификатор, названия классов и параметры preprocessing в один файл."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "id2label": self.id2label,
            "clean_kwargs": self.clean_kwargs,
            "metadata": self.metadata,
            "sklearn_version": sklearn.__version__,
        }, path)

    @classmethod
    def load(cls, path):
        """Загружает модель. Открывайте только свои файлы: joblib/pickle может выполнить код из файла."""
        data = joblib.load(path)
        if data["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"Модель обучена в scikit-learn {data['sklearn_version']}, а установлен {sklearn.__version__}: "
                "результаты могут отличаться."
            )
        return cls(data["vectorizer"], data["classifier"], data["id2label"],
                   data["clean_kwargs"], data["metadata"])

    def predict(self, texts):
        """Классифицирует один текст (str) или список текстов"""
        if isinstance(texts, str):
            texts = [texts]
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("predict ожидает строку или список строк")

        cleaned = [clean_text(t, **self.clean_kwargs) for t in texts]
        valid = [i for i, t in enumerate(cleaned) if t]

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
