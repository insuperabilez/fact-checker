import json
import pandas as pd

def process_fever_to_csv(input_file, output_file):
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            claim = entry['claim']
            label = entry['label']
            
            # Преобразуем метки: REFUTES -> 0, SUPPORTS -> 1, остальные пропускаем
            if label == 'REFUTES':
                data.append({'text': claim, 'label': 0})
            elif label == 'SUPPORTS':
                data.append({'text': claim, 'label': 1})
            # NOT ENOUGH INFO и другие метки игнорируем
    
    # Создаем DataFrame и сохраняем в CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')

# Пример вызова функции
process_fever_to_csv('train.jsonl', 'fever_train_processed.csv')