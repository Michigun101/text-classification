# Классификация новостей по темам

Мини-проект по классификации текстов: от данных до предсказания. По тексту английской новости модель определяет её тему:
**World**, **Sports**, **Business** или **Sci/Tech**. Классический ML (TF-IDF + логистическая регрессия), без нейросетей и GPU.

| | |
|---|---|
| Задача | многоклассовая классификация текста, 4 класса, один класс на новость |
| Метрика | macro F1 (классы сбалансированы, поэтому accuracy почти совпадает) |
| Результат | **macro F1 0.8506, accuracy 0.8525** на отложенной выборке из 400 новостей |
| Baseline | TF-IDF + LogisticRegression: macro F1 0.8398 |

## Датасет

Выборка из открытого **AG News** (новости, 4 темы). Скрипт `prepare_dataset.py` скачивает её и создаёт `data/dataset.csv`:
2000 новостей, по 500 на каждый класс, `random_state=42`.

| Колонка | Тип | Описание |
|---|---|---|
| `text` | str | заголовок и текст новости (английский) |
| `label` | int | номер класса: 0 — World, 1 — Sports, 2 — Business, 3 — Sci/Tech |
| `label_name` | str | название класса |

Средняя длина — 38 слов. Данные «грязные» (HTML-сущности, подписи агентств): их чистит `textclf/preprocessing.py`.
Файл `data/dataset.csv` в git не хранится, его создаёт скрипт.

## Структура проекта

```
text-classification/
├── textclf/                  основной код (пакет)
│   ├── preprocessing.py      очистка текста, метки, подготовка датасета
│   └── model.py              pipeline модели и класс TextClassifier (save / load / predict)
├── prepare_dataset.py        скачивание датасета -> data/dataset.csv
├── train.py                  обучение + оценка + сохранение модели
├── predict.py                inference (CLI)
├── examples/demo_texts.txt   demo-примеры для predict.py --demo
├── results/                  метрики, confusion matrix, анализ ошибок, таблица экспериментов
├── days/                     скрипты по дням (EDA, baseline, эксперименты, error analysis)
├── docs/development-log.md   подробный журнал разработки по дням
├── models/                   сохранённая модель (создаётся train.py, в git не хранится)
└── requirements.txt
```

## Установка

Нужен Python 3.11+ (проверено на 3.14).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск обучения

```bash
python prepare_dataset.py   # скачивает AG News, создаёт data/dataset.csv (нужен интернет)
python train.py             # обучает модель (несколько секунд), печатает метрики
```

`train.py` очищает тексты, делит данные 80/20 (`stratify`, `random_state=42`: 1599 train / 400 test), обучает модель,
считает метрики на test и сохраняет:

- `models/text_classifier.joblib` — модель (vectorizer, классификатор, названия классов, параметры очистки);
- `results/final_metrics.json` и `results/final_report.txt` — метрики и отчёт по классам.

Параметры: `python train.py --data путь/к/файлу.csv --model куда/сохранить.joblib`.

## Запуск inference

Сначала нужна обученная модель (`python train.py`). Переобучать ничего не надо.

```bash
python predict.py "Oil prices jump as OPEC announces surprise production cuts"
python predict.py --demo                 # примеры из examples/demo_texts.txt
python predict.py --file мои_тексты.txt  # по одному тексту в строке
python predict.py "some text" --json     # полный ответ в JSON
```

Вывод — тема и уверенность модели:

```
[Business 0.71] Oil prices jump as OPEC announces surprise production cuts
```

Из Python:

```python
from textclf import TextClassifier

clf = TextClassifier.load("models/text_classifier.joblib")
clf.predict("Oil prices jump as OPEC announces surprise production cuts")
```

**Вход:** строка или список строк (сырой текст, очищать заранее не нужно: `predict` сам применяет ту же очистку, что при обучении).
**Выход:** список словарей, по одному на текст:

```json
{
  "text": "Oil prices jump as OPEC announces surprise production cuts",
  "label": "Business",
  "label_id": 2,
  "confidence": 0.7148,
  "probabilities": {"World": 0.1015, "Sports": 0.0851, "Business": 0.7148, "Sci/Tech": 0.0986},
  "error": null
}
```

Если после очистки текст пуст (`""`, пробелы, одни теги), `label`, `confidence` и т.д. равны `null`, а в `error` написана причина.
Если вместо строк передать что-то другое, будет `TypeError`.

## Demo-примеры

Результат `python predict.py --demo` (10 текстов из `examples/demo_texts.txt`; девятый — «грязный» текст с HTML-сущностями и подписью Reuters):

| Текст | Предсказание | Уверенность |
|---|---|---|
| Manchester United beat Chelsea 2-1 in a thrilling Premier League match… | Sports | 0.76 |
| Federer wins the Wimbledon final after a five-set battle | Sports | 0.53 |
| Oil prices jump as OPEC announces surprise production cuts… | Business | 0.76 |
| The central bank raised interest rates for the third time this year… | Business | 0.59 |
| Prime minister meets president to discuss peace talks in the Middle East | World | 0.90 |
| UN Security Council votes on sanctions after the border clashes | World | 0.36 |
| Apple unveils new iPhone with faster chip and improved camera | Sci/Tech | 0.60 |
| Scientists discover water on a distant planet using the new space telescope | Sci/Tech | 0.78 |
| LONDON (Reuters) - Shares rose as the company #39;s profit beat forecasts… | Business | 0.66 |
| Microsoft agrees to buy a software maker for $2 billion | Sci/Tech | 0.60 |

Все ответы совпадают с ожидаемой темой. Но заметьте низкую уверенность в «UN Security Council…» (0.36): модель права, но «не уверена».

## Метрики

Все метрики посчитаны на одной и той же отложенной выборке из 400 новостей (по 100 на класс).

| Модель | Accuracy | Macro F1 |
|---|---|---|
| Baseline: TF-IDF (слова и пары слов) + LogisticRegression | 0.8425 | 0.8398 |
| **Лучшая: то же + удаление стоп-слов** | **0.8525** | **0.8506** |

Лучшая модель по классам:

| Класс | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.8411 | 0.9000 | 0.8696 |
| Sports | 0.8952 | 0.9400 | 0.9171 |
| Business | 0.8372 | 0.7200 | 0.7742 |
| Sci/Tech | 0.8333 | 0.8500 | 0.8416 |

**Погрешность.** На 400 примерах доверительный интервал accuracy — около ±0.018 (95%), а разброс кросс-валидации между фолдами 0.01–0.016. Поэтому разницу baseline и лучшей модели (+0.011) нельзя считать надёжно доказанной.

## Baseline, улучшения и выбор лучшей модели

**Baseline** (`days/classification_day03.py`): `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` + `LogisticRegression`. Macro F1 0.8398.

**Улучшения** (`days/classification_day04.py`, таблица в `results/day04_experiments.csv`). Варианты сравнивались 5-кратной кросс-валидацией на train, лучший выбирался по ней, а не по test:

| Что пробовали | CV macro F1 | Вывод |
|---|---|---|
| baseline | 0.8379 | — |
| только слова (1,1) / до троек (1,3) | 0.8420 / 0.8367 | не помогло |
| **удаление английских стоп-слов** | **0.8530** | лучший, и самое простое изменение |
| символьные n-граммы + слова | 0.8515 | чуть лучше baseline, в 6 раз медленнее |
| ComplementNB вместо LR | 0.8517 | то же качество, очень быстро |
| LinearSVC, большие `C`, GridSearchCV | 0.834–0.851 | не лучше |
| CountVectorizer вместо TF-IDF | 0.8044 | хуже |

Выбран TF-IDF + стоп-слова + LogisticRegression: максимальный CV, минимальное усложнение. Топ-4 варианта статистически не различимы; потолок настройки TF-IDF — около 0.85.

## Error analysis (кратко)

Подробно: `days/classification_day05.py`, `results/error_analysis.txt`, `results/errors.csv`, `results/confusion_matrix.png`.

Ошибок 59 из 400 (14.8%). Больше всего их у **Business: 28 из 100** (в Sci/Tech 15, в World 11).

- **Business ↔ World:** финансы с политикой («Pinochet bank accounts», «Russia foreign debt»): слова `president`, `minister` тянут к World.
- **Business ↔ Sci/Tech:** корпоративные новости технологических компаний. Часть «ошибок» — спорные ярлыки.
- **Многозначные слова:** «model train maker» → Sci/Tech. TF-IDF не видит контекст.
- **Короткие тексты:** точность 0.80 на самых коротких 25% (≤ 31 слова) против 0.87 на остальных.
- **Уверенность информативна:** при `confidence > 0.5` (54% текстов) точность около 94%; при ≤ 0.5 — 0.75.
- **Артефакт очистки:** в 18 исходных текстах остался фрагмент `amp;` (`AT amp;T`) — известная недоработка `clean_text`.

## Ограничения модели

1. **Только 4 темы и один ответ.** Модель не знает других рубрик: новость про кино, еду или здоровье всё равно получит одну из четырёх тем. Проверка на тексте вне домена: рецепт торта → Sports (0.36), «The film won the Oscar for best picture» → Sports (0.30). Низкая уверенность (< 0.4) есть сигнал «вне домена», но специальной проверки нет.
2. **Только английский.** Русский текст «Президент подписал новый закон о налогах» получил Sci/Tech с уверенностью 0.27: слова не из словаря модели.
3. **Точность около 85%.** Каждая седьмая новость классифицируется неверно; для Business — каждая четвёртая (recall 0.72). Для задач, где ошибка дорого стоит, нужна ручная проверка предсказаний с `confidence < 0.5`.
4. **Не понимает контекст.** TF-IDF считает слова «мешком»: не различает смыслы слова («model»), не понимает отрицание и порядок слов.
5. **Устаревшие и узкие данные.** Обучение на 1599 новостях из AG News (новости ~2004-2005 гг. из ограниченного набора источников): новые термины и события модель не знает.
6. **`confidence` — не калиброванная вероятность.** Это выход логистической регрессии; он полезен для ранжирования, но 0.76 не значит «верно в 76% случаев».
7. **Оценка на малой выборке.** 400 тестовых примеров дают погрешность ±1.8 п.п., а разметка AG News местами спорная.
8. **Очень короткие тексты.** Предсказание для одного слова (`"ok"`) возвращается, но с низкой уверенностью (0.32). Тексты, которые пусты после очистки, получают `error`.
9. **Технические.** Модель привязана к версии scikit-learn (при несовпадении — предупреждение); `.joblib` — это pickle, открывайте только свои файлы. Подписи агентств и HTML-мусор чистятся под формат AG News: на данных из другого источника очистку нужно перепроверить.

## Что можно улучшить дальше

- Контекстные модели (эмбеддинги трансформеров, дообучение): основная надежда на Business / World / Sci/Tech и многозначные слова.
- Больше данных (датасет здесь — лишь 2000 из ~120 000 доступных) и ревизия спорных ярлыков.
- Порог уверенности и отдельный ответ «не знаю» для вне-доменных текстов.
- Исправить `amp;` в `clean_text` и переобучить.

## Журнал разработки

Скрипты по дням лежат в `days/` (запускаются из корня репозитория, например `python days/classification_day01.py`), подробный журнал с разбором кода — в `docs/development-log.md`.

| День | Что сделано | Скрипт |
|---|---|---|
| 1 | загрузка датасета, EDA | `days/classification_day01.py` |
| 2 | очистка текста, метки | `days/classification_day02.py` |
| 3 | baseline | `days/classification_day03.py` |
| 4 | эксперименты по улучшению | `days/classification_day04.py` |
| 5 | error analysis | `days/classification_day05.py` |
| 6 | inference pipeline | `days/classification_day06.py` |
| 7 | финальная сборка проекта | `train.py`, `predict.py`, `textclf/` |
