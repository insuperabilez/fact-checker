import os
import pandas as pd

def merge_csv_files(output_filename='train.csv'):
    # Получаем список всех CSV файлов в текущей директории
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != output_filename]
    
    if not csv_files:
        print("В директории нет CSV файлов для объединения.")
        return
    
    # Читаем и объединяем все CSV файлы
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Добавляем столбец с именем файла
        df['source'] = csv_file
        dfs.append(df)
        print(f"Добавлен файл: {csv_file} (строк: {len(df)})")
    
    # Объединяем все DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Сохраняем результат
    merged_df.to_csv(output_filename, index=False)
    print(f"\nИтоговый файл '{output_filename}' создан. Всего строк: {len(merged_df)}")
    print(f"Добавлен столбец 'source' с именами исходных файлов")

if __name__ == "__main__":
    merge_csv_files()