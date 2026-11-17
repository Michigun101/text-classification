"""textclf — классификация новостных текстов: preprocessing, модель и inference."""
from textclf.model import TextClassifier, build_pipeline
from textclf.preprocessing import clean_text, preprocess_dataframe

__all__ = ["TextClassifier", "build_pipeline", "clean_text", "preprocess_dataframe"]
