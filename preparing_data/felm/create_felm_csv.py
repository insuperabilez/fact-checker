from datasets import load_dataset
import pandas as pd

# Все доступные конфигурации датасета
configs = ['math', 'reasoning', 'science', 'wk', 'writing_rec']

# Список для хранения всех пар (text, label)
all_pairs = []

for config in configs:
    print(f"Обработка конфигурации: {config}")
    
    try:
        # Загрузка датасета для текущей конфигурации
        dataset = load_dataset('hkust-nlp/felm', config)
        
        # Обработка всех подмножеств (train/validation/test)
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                print(f"  Обработка {split}...")
                for example in dataset[split]:
                    # Создание пар (сегмент, метка) для каждого примера
                    for segment, label in zip(example['segmented_response'], example['labels']):
                        all_pairs.append({
                            'text': segment,
                            'label': int(label),  # Преобразуем в int для единообразия
                            'config': config,    # Добавляем информацию о конфигурации
                            'source': example.get('source', ''),
                            'error_type': example.get('type', [''])[0] if label == 0 else ''
                        })
    except Exception as e:
        print(f"Ошибка при обработке {config}: {str(e)}")
        continue

# Создание DataFrame
df = pd.DataFrame(all_pairs)
df = df[['text','label']]
# Сохранение в CSV
output_file = 'felm_all_configs_text_label.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nГотово! Результат сохранён в {output_file}")
print(f"Всего обработано {len(df)} сегментов.")
print(f"Распределение меток:\n{df['label'].value_counts()}")
print(f"Распределение по конфигурациям:\n{df['config'].value_counts()}")