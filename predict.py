"""Inference: классификация новых текстов обученной моделью """
import argparse
import json

from textclf import TextClassifier

DEMO_FILE = "examples/demo_texts.txt"


def main():
    parser = argparse.ArgumentParser(description="Классификация новостных текстов")
    parser.add_argument("texts", nargs="*", help="тексты для классификации (каждый в кавычках)")
    parser.add_argument("--file", help="файл с текстами, по одному в строке")
    parser.add_argument("--demo", action="store_true", help=f"классифицировать примеры из {DEMO_FILE}")
    parser.add_argument("--model", default="models/text_classifier.joblib", help="путь к модели")
    parser.add_argument("--json", action="store_true", help="вывести полный JSON вместо краткой таблицы")
    args = parser.parse_args()

    texts = list(args.texts)
    path = DEMO_FILE if args.demo else args.file
    if path:
        with open(path, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        parser.error("нужен хотя бы один текст: передайте его аргументом, --file или --demo")

    try:
        clf = TextClassifier.load(args.model)
    except FileNotFoundError:
        raise SystemExit(f"Модель {args.model} не найдена. Сначала обучите её: python train.py")

    results = clf.predict(texts)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for r in results:
        if r["error"]:
            print(f"[{'-':8s} {'-':>4s}] {r['text'][:80]!r} ({r['error']})")
        else:
            print(f"[{r['label']:8s} {r['confidence']:.2f}] {r['text'][:90]}")


if __name__ == "__main__":
    main()
