import pandas as pd

# Загрузка всех TSV-файлов
train = pd.read_csv("train.tsv", sep="\t", header=None)
test = pd.read_csv("test.tsv", sep="\t", header=None)
valid = pd.read_csv("valid.tsv", sep="\t", header=None)

# Объединение данных
combined = pd.concat([train, test, valid], ignore_index=True)

# Назначение названий колонок (согласно описанию формата LIAR)
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]
combined.columns = columns

# Фильтрация и преобразование меток:
# 1 = "true", 0 = "false" или "pants-fire"
combined["binary_label"] = combined["label"].apply(
    lambda x: 0 if x == "true" else 1 if x in ['false', 'pants-fire'] else None
)

combined = combined.dropna(subset='binary_label')

result = combined[["statement", "binary_label"]]

# Переименование столбцов
result.columns = ["text", "label"]

# Сохранение в CSV
result.to_csv("liar_binary.csv", index=False, encoding="utf-8")

print("Готово! Результат сохранён в liar_binary.csv")