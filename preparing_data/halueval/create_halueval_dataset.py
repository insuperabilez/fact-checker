import json
import pandas as pd

# 1. Обработка qa_data.json (формат JSON Lines)
processed_qa = []
with open('qa_data.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            # Объект 1: knowledge + right_answer (label=1)
            processed_qa.append({
                'text': str(item.get('knowledge', '')) + ' ' + str(item.get('right_answer', '')),
                'label': 1
            })
            # Объект 2: knowledge + hallucinated_answer (label=0)
            processed_qa.append({
                'text': str(item.get('knowledge', '')) + ' ' + str(item.get('hallucinated_answer', '')),
                'label': 0
            })
        except json.JSONDecodeError as e:
            print(f"Ошибка при обработке строки: {line}. Пропускаем...")

# Сохранение в CSV
pd.DataFrame(processed_qa).to_csv('qa_processed.csv', index=False)

# 2. Обработка general_data.json (формат JSON Lines)
processed_general = []
with open('general_data.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            label = 0 if str(item.get('hallucination', '')).lower() == 'yes' else 1
            processed_general.append({
                'text': str(item.get('user_query', '')) + ' ' + str(item.get('chatgpt_response', '')),
                'label': label
            })
        except json.JSONDecodeError as e:
            print(f"Ошибка при обработке строки: {line}. Пропускаем...")

# Сохранение в CSV
pd.DataFrame(processed_general).to_csv('general_processed.csv', index=False)

