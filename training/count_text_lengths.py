import pandas as pd
from transformers import AutoTokenizer

# Загрузка модели и токенизатора ModernBERT
MODEL_NAME = "answerdotai/ModernBERT-base"  # или укажите конкретную версию ModernBERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Загрузка данных
df = pd.read_csv("train.csv")

# Функция для подсчета токенов
def count_tokens(text):
    if pd.isna(text):
        return 0
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text, truncation=True)))
    return len(tokens)

# Применяем функцию к столбцу text и создаем новый столбец
print("Вычисление длины текстов в токенах...")
from tqdm import tqdm
tqdm.pandas()
df['token_length'] = df['text'].progress_apply(count_tokens)

# Сохраняем результат
df.to_csv("train_with_token_length.csv", index=False)

# Общая статистика
print("\nОбщая статистика:")
print(f"Всего строк: {len(df)}")
print(f"Максимальная длина в токенах: {df['token_length'].max()}")
print(f"Минимальная длина в токенах: {df['token_length'].min()}")
print(f"Средняя длина в токенах: {df['token_length'].mean():.1f}")

# Статистика по источникам (столбец source)
if 'source' in df.columns:
    print("\nСтатистика по источникам:")
    source_stats = df.groupby('source')['token_length'].agg(['count', 'mean', 'min', 'max'])
    source_stats.columns = ['Количество строк', 'Средняя длина', 'Минимальная длина', 'Максимальная длина']
    print(source_stats.round(1))
    
    # Сохраняем статистику по источникам в отдельный файл
    source_stats.to_csv("token_length_by_source.csv")
    print("\nСтатистика по источникам сохранена в token_length_by_source.csv")
else:
    print("\nСтолбец 'source' не найден в данных")

print("\nРезультат с длинами токенов сохранен в train_with_token_length.csv")