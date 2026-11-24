import pandas as pd
import numpy as np
import os
import time
import json
from pandas.io.json import json_normalize

print("=" * 60)
print(" ЗАДАЧА 1: Создание DataFrame для сохранения во всех форматах")
print("=" * 60)

df = pd.DataFrame({
    'id': range(1, 101),
    'name': [f'Сотрудник_{i}' for i in range(1, 101)],
    'age': np.random.randint(22, 60, size=100),
    'salary': np.random.randint(40000, 150000, size=100),
    'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], size=100),
    'hiredate': pd.date_range(start='2015-01-01', periods=100, freq='D'),
    'is_active': np.random.choice([True, False], size=100, p=[0.85, 0.15])
})

print(f"\n[1.1] DataFrame создан:")
print(df.head(10))
print(f"\nФорма: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 2: Работа с CSV")
print("=" * 60)

# 2.1 — Сохранить в CSV двумя способами
df.to_csv('data_with_index.csv', index=True)
df.to_csv('data_no_index.csv', index=False)
print(f"\n[2.1] Сохранено:")
print(f"      data_with_index.csv (с индексом)")
print(f"      data_no_index.csv (без индекса)")

df_read = pd.read_csv('data_with_index.csv')
print(f"\n[2.2] Прочитано из CSV с индексом:")
print(f"      Типы данных:\n{df_read.dtypes}")
print(f"\nПроблема: hiredate стал объектом (str), а не datetime!")
print(f"      is_active стал объектом (str), а не bool!")

size_with = os.path.getsize('data_with_index.csv')
size_without = os.path.getsize('data_no_index.csv')
print(f"\n[2.3] Размеры файлов:")
print(f"      С индексом:    {size_with} байт ({size_with/1024:.2f} KB)")
print(f"      Без индекса:   {size_without} байт ({size_without/1024:.2f} KB)")
print(f"      Разница: {size_with - size_without} байт")

print(f"\n[2.4] Оптимизированное чтение CSV:")

t1 = time.time()
df_slow = pd.read_csv('data_no_index.csv')
t_slow = time.time() - t1
print(f"      Обычное чтение: {t_slow:.4f} сек")

t2 = time.time()
df_fast = pd.read_csv(
    'data_no_index.csv',
    dtype={'id': 'int32', 'age': 'int32', 'salary': 'int32', 'department': 'category'},
    parse_dates=['hiredate'],
    usecols=['id', 'name', 'salary', 'department', 'hiredate']
)
t_fast = time.time() - t2
print(f"      Оптимизированное чтение: {t_fast:.4f} сек")
print(f"      Ускорение: в {t_slow/t_fast:.1f} раз")
print(f"\n      Типы данных после оптимизации:\n{df_fast.dtypes}")
print(f"      Форма: {df_fast.shape} (меньше столбцов — только нужные)")


print("\n" + "=" * 60)
print(" ЗАДАЧА 3: Работа с JSON")
print("=" * 60)

df_skills = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Анна', 'Борис', 'Виктор', 'Галина', 'Дмитрий'],
    'skills': [
        ['Python', 'Pandas', 'NumPy'],
        ['Java', 'Spring', 'SQL'],
        ['Python', 'Django', 'PostgreSQL', 'Docker'],
        ['R', 'Statistics', 'Python'],
        ['JavaScript', 'React', 'Node.js']
    ],
    'experience': [5, 8, 3, 10, 4]
})
print(f"\n[3.1] DataFrame с вложенным столбцом 'skills':\n{df_skills}")

df_skills.to_json('skills_records.json', orient='records', force_ascii=False, indent=2)
df_skills.to_json('skills_columns.json', orient='columns', force_ascii=False, indent=2)
df_skills.to_json('skills_index.json', orient='index', force_ascii=False, indent=2)
df_skills.to_json('skills_split.json', orient='split', force_ascii=False, indent=2)

print(f"\n[3.2] Сохранено в 4 ориентациях:")
print(f"      skills_records.json — ориентация 'records'")
print(f"      skills_columns.json — ориентация 'columns'")
print(f"      skills_index.json   — ориентация 'index'")
print(f"      skills_split.json   — ориентация 'split'")

print(f"\n[3.3] Сравнение размеров и структуры:")
for fname in ['skills_records.json', 'skills_columns.json', 'skills_index.json', 'skills_split.json']:
    size = os.path.getsize(fname)
    print(f"      {fname:25s} — {size} байт")

print(f"\nПример структуры 'records':")
with open('skills_records.json', 'r', encoding='utf-8') as f:
    print(f.read()[:300])

print(f"\nПример структуры 'columns':")
with open('skills_columns.json', 'r', encoding='utf-8') as f:
    print(f.read()[:300])

df_from_json = pd.read_json('skills_records.json')
print(f"\n[3.4] Прочитано из JSON:\n{df_from_json}")

nested_data = [
    {
        'id': 1,
        'name': 'Анна',
        'skills': [
            {'name': 'Python', 'level': 'advanced'},
            {'name': 'Pandas', 'level': 'intermediate'}
        ]
    },
    {
        'id': 2,
        'name': 'Борис',
        'skills': [
            {'name': 'Java', 'level': 'advanced'},
            {'name': 'SQL', 'level': 'beginner'}
        ]
    }
]

df_normalized = json_normalize(
    nested_data,
    record_path='skills',
    meta=['id', 'name'],
    record_prefix='skill_'
)
print(f"\nРезультат json_normalize (разворачивание вложенности):")
print(df_normalized)


print("\n" + "=" * 60)
print(" ЗАДАЧА 4: Parquet")
print("=" * 60)

try:
    import pyarrow
except ImportError:
    print("\n[!] Установите pyarrow: pip install pyarrow")
    print("    Пропускаем задачу 4...\n")
else:
    df.to_parquet('data_snappy.parquet', compression='snappy')
    df.to_parquet('data_gzip.parquet', compression='gzip')
    df.to_parquet('data_brotli.parquet', compression='brotli')
    print(f"\n[4.1] Сохранено в Parquet с разными сжатиями:")
    print(f"      data_snappy.parquet")
    print(f"      data_gzip.parquet")
    print(f"      data_brotli.parquet")

    size_csv = os.path.getsize('data_no_index.csv')
    size_snappy = os.path.getsize('data_snappy.parquet')
    size_gzip = os.path.getsize('data_gzip.parquet')
    size_brotli = os.path.getsize('data_brotli.parquet')

    print(f"\n[4.2] Сравнение размеров файлов:")
    print(f"      CSV:              {size_csv:>10} байт ({size_csv/1024:.2f} KB)")
    print(f"      Parquet (snappy): {size_snappy:>10} байт ({size_snappy/1024:.2f} KB)")
    print(f"      Parquet (gzip):   {size_gzip:>10} байт ({size_gzip/1024:.2f} KB)")
    print(f"      Parquet (brotli): {size_brotli:>10} байт ({size_brotli/1024:.2f} KB)")
    print(f"\n      Parquet (snappy) меньше CSV в {size_csv/size_snappy:.1f} раз!")

    t_csv = time.time()
    pd.read_csv('data_no_index.csv')
    t_csv = time.time() - t_csv

    t_parquet = time.time()
    pd.read_parquet('data_snappy.parquet')
    t_parquet = time.time() - t_parquet

    print(f"\n[4.3] Время чтения:")
    print(f"      CSV:     {t_csv:.4f} сек")
    print(f"      Parquet: {t_parquet:.4f} сек")
    print(f"      Parquet быстрее в {t_csv/t_parquet:.1f} раз")

    t_all = time.time()
    pd.read_parquet('data_snappy.parquet')
    t_all = time.time() - t_all

    t_pruned = time.time()
    pd.read_parquet('data_snappy.parquet', columns=['name', 'salary'])
    t_pruned = time.time() - t_pruned

    print(f"\n[4.4] Column pruning (чтение только нужных столбцов):")
    print(f"      Все столбцы:          {t_all:.4f} сек")
    print(f"      Только name, salary:  {t_pruned:.4f} сек")
    print(f"      Ускорение: в {t_all/t_pruned:.1f} раз")
    print(f"      (CSV так не умеет — читает весь файл)")

    df_parquet = pd.read_parquet('data_snappy.parquet')
    print(f"\n[4.5] Типы данных после чтения Parquet:")
    print(df_parquet.dtypes)
    print(f"\n      ✅ hiredate остался datetime64[ns]")
    print(f"      ✅ is_active остался bool")
    print(f"      ✅ department остался category (если был category)")
    print(f"      Parquet сохраняет типы данных, CSV — нет!")


print("\n" + "=" * 60)
print(" ЗАДАЧА 5: HDF5")
print("=" * 60)

df_agg = df.groupby('department').agg({
    'salary': ['mean', 'max', 'min'],
    'age': 'mean'
}).round(2)
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns]

with pd.HDFStore('data.h5', mode='w') as store:
    store['employees'] = df
    store['department_stats'] = df_agg

print(f"\n[5.1] Сохранено в HDF5:")
print(f"      data.h5")
print(f"      Ключ 'employees' — основной DataFrame")
print(f"      Ключ 'department_stats' — агрегированный DataFrame")

df_from_h5 = pd.read_hdf('data.h5', key='employees')
print(f"\n[5.2] Прочитано по ключу 'employees':")
print(df_from_h5.head())

df_stats = pd.read_hdf('data.h5', key='department_stats')
print(f"\nПрочитано по ключу 'department_stats':\n{df_stats}")

df.to_hdf('data_fixed.h5', key='data', format='fixed')
df.to_hdf('data_table.h5', key='data', format='table')

size_fixed = os.path.getsize('data_fixed.h5')
size_table = os.path.getsize('data_table.h5')

print(f"\n[5.3] Сравнение форматов fixed и table:")
print(f"      fixed: {size_fixed} байт ({size_fixed/1024:.2f} KB)")
print(f"      table: {size_table} байт ({size_table/1024:.2f} KB)")
print(f"      table поддерживает запросы через where, fixed — нет")

# 5.4 — Запрос к таблице HDF5 без загрузки всего файла
print(f"\n[5.4] Запрос к HDF5 без загрузки всего файла:")
result = pd.read_hdf('data_table.h5', key='data',
                     where='department == "IT" and salary > 80000')
print(f"      Сотрудники IT с зарплатой > 80000:")
print(result[['name', 'department', 'salary']].head(10))
print(f"\n      Загружено только {len(result)} строк из 100")

size_h5 = os.path.getsize('data.h5')
size_csv = os.path.getsize('data_no_index.csv')

print(f"\n[5.5] Сравнение размеров:")
print(f"      CSV:  {size_csv} байт ({size_csv/1024:.2f} KB)")
print(f"      HDF5: {size_h5} байт ({size_h5/1024:.2f} KB)")
print(f"      HDF5 меньше CSV в {size_csv/size_h5:.1f} раз")


print("\n" + "=" * 60)
print(" ВСЕ ЗАДАЧИ ПО ФОРМАТАМ ХРАНЕНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)

print("\n💡 Созданные файлы:")
for f in os.listdir('.'):
    if f.endswith(('.csv', '.json', '.parquet', '.h5')):
        size = os.path.getsize(f)
        print(f"   {f:30s} — {size:>10} байт ({size/1024:.2f} KB)")

print("\n" + "=" * 60)
print(" ЗАДАЧА 6: Сводная таблица метрик и выводы")
print("=" * 60)

# 6.1 — Сбор метрик: размер файла, время записи, время чтения

metrics = []

# CSV
size_csv = os.path.getsize('data_no_index.csv')
t_write_csv = time.time()
df.to_csv('data_no_index.csv', index=False)
t_write_csv = time.time() - t_write_csv

t_read_csv = time.time()
pd.read_csv('data_no_index.csv')
t_read_csv = time.time() - t_read_csv

metrics.append({
    'format': 'CSV',
    'file_size_bytes': size_csv,
    'file_size_kb': round(size_csv / 1024, 2),
    'write_time_sec': round(t_write_csv, 4),
    'read_time_sec': round(t_read_csv, 4),
    'preserves_types': False,
    'supports_compression': False,
    'supports_column_pruning': False,
    'supports_multiple_objects': False,
    'human_readable': True
})

# JSON
size_json = os.path.getsize('skills_records.json')
t_write_json = time.time()
df_skills.to_json('skills_records.json', orient='records', force_ascii=False)
t_write_json = time.time() - t_write_json

t_read_json = time.time()
pd.read_json('skills_records.json')
t_read_json = time.time() - t_read_json

metrics.append({
    'format': 'JSON',
    'file_size_bytes': size_json,
    'file_size_kb': round(size_json / 1024, 2),
    'write_time_sec': round(t_write_json, 4),
    'read_time_sec': round(t_read_json, 4),
    'preserves_types': False,
    'supports_compression': False,
    'supports_column_pruning': False,
    'supports_multiple_objects': False,
    'human_readable': True
})

# Parquet (snappy)
size_parquet = os.path.getsize('data_snappy.parquet')
t_write_parquet = time.time()
df.to_parquet('data_snappy.parquet', compression='snappy')
t_write_parquet = time.time() - t_write_parquet

t_read_parquet = time.time()
pd.read_parquet('data_snappy.parquet')
t_read_parquet = time.time() - t_read_parquet

metrics.append({
    'format': 'Parquet (snappy)',
    'file_size_bytes': size_parquet,
    'file_size_kb': round(size_parquet / 1024, 2),
    'write_time_sec': round(t_write_parquet, 4),
    'read_time_sec': round(t_read_parquet, 4),
    'preserves_types': True,
    'supports_compression': True,
    'supports_column_pruning': True,
    'supports_multiple_objects': False,
    'human_readable': False
})

# HDF5 (table format)
size_hdf5 = os.path.getsize('data_table.h5')
t_write_hdf5 = time.time()
df.to_hdf('data_table.h5', key='data', format='table', mode='w')
t_write_hdf5 = time.time() - t_write_hdf5

t_read_hdf5 = time.time()
pd.read_hdf('data_table.h5', key='data')
t_read_hdf5 = time.time() - t_read_hdf5

metrics.append({
    'format': 'HDF5 (table)',
    'file_size_bytes': size_hdf5,
    'file_size_kb': round(size_hdf5 / 1024, 2),
    'write_time_sec': round(t_write_hdf5, 4),
    'read_time_sec': round(t_read_hdf5, 4),
    'preserves_types': True,
    'supports_compression': True,
    'supports_column_pruning': True,
    'supports_multiple_objects': True,
    'human_readable': False
})

# 6.2 — Построить итоговый DataFrame-отчёт
df_report = pd.DataFrame(metrics)

print(f"\n[6.1] Метрики собраны для 4 форматов")
print(f"\n[6.2] Итоговый DataFrame-отчёт:")
print(df_report.to_string(index=False))

# Нормализация для сравнения
df_report_norm = df_report.copy()
df_report_norm['file_size_kb_norm'] = df_report_norm['file_size_kb'] / df_report_norm['file_size_kb'].max()
df_report_norm['write_time_norm'] = df_report_norm['write_time_sec'] / df_report_norm['write_time_sec'].max()
df_report_norm['read_time_norm'] = df_report_norm['read_time_sec'] / df_report_norm['read_time_sec'].max()

print(f"\nНормализованные метрики (1.0 = худший результат):")
print(df_report_norm[['format', 'file_size_kb_norm', 'write_time_norm', 'read_time_norm']].to_string(index=False))

# 6.3 — Выводы и рекомендации
print(f"\n[6.3] Выводы и рекомендации:")
print(f"\n{'='*60}")
print(f" КАКОЙ ФОРМАТ ВЫБРАТЬ?")
print(f"{'='*60}")

print(f"\n📊 Для аналитики и Data Science:")
print(f"   ✅ Parquet — лучший выбор")
print(f"   • Быстрое чтение/запись")
print(f"   • Маленький размер файла")
print(f"   • Сохраняет типы данных")
print(f"   • Поддерживает column pruning")
print(f"   • Интеграция с PySpark, Dask, Pandas")

print(f"\n🔄 Для обмена данными между системами:")
print(f"   ✅ CSV или JSON")
print(f"   • CSV — простой, читаемый, универсальный")
print(f"   • JSON — для вложенных структур, API")
print(f"   • Поддерживается всеми языками и инструментами")
print(f"   • Можно открыть в Excel, текстовом редакторе")

print(f"\n🔬 Для научных расчётов и больших данных:")
print(f"   ✅ HDF5 или Parquet")
print(f"   • HDF5 — для многомерных массивов, научных данных")
print(f"   • Поддерживает несколько объектов в одном файле")
print(f"   • Быстрый произвольный доступ")
print(f"   • Parquet — для табличных данных, аналитики")

print(f"\n💾 Для архивирования и хранения:")
print(f"   ✅ Parquet (gzip/brotli) или HDF5")
print(f"   • Максимальное сжатие")
print(f"   • Сохранение метаданных и типов")

print(f"\n{'='*60}")
print(f" СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print(f"{'='*60}")

comparison = pd.DataFrame({
    'Критерий': ['Размер файла', 'Скорость записи', 'Скорость чтения',
                 'Сохранение типов', 'Сжатие', 'Column pruning',
                 'Несколько объектов', 'Читаемость человеком', 'Универсальность'],
    'CSV': ['❌ Большой', '⚡ Средне', '⚡ Средне',
            '❌ Нет', '❌ Нет', '❌ Нет',
            '❌ Нет', '✅ Да', '✅ Отличная'],
    'JSON': ['❌ Большой', '⚡ Средне', '⚡ Средне',
             '❌ Нет', '❌ Нет', '❌ Нет',
             '❌ Нет', '✅ Да', '✅ Отличная'],
    'Parquet': ['✅ Маленький', '🚀 Быстро', '🚀 Быстро',
                '✅ Да', '✅ Да', '✅ Да',
                '❌ Нет', '❌ Нет', '⚡ Хорошая'],
    'HDF5': ['⚡ Средний', '🚀 Быстро', '🚀 Быстро',
             '✅ Да', '✅ Да', '✅ Да',
             '✅ Да', '❌ Нет', '⚡ Хорошая']
})

print(comparison.to_string(index=False))

print(f"\n{'='*60}")
print(f" ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
print(f"{'='*60}")

print(f"""
1. Если работаешь с Pandas и нужна скорость → Parquet
2. Если нужно передать данные коллеге → CSV
3. Если данные вложенные (API, JSON) → JSON
4. Если хранишь много разных таблиц → HDF5
5. Если данные для научных расчётов → HDF5 или Parquet
6. Если нужна максимальная совместимость → CSV

Оптимальный стек для Data Science:
  • Хранение: Parquet
  • Обмен: CSV/JSON
  • Архив: Parquet (gzip) или HDF5
""")

print(f"\n{'='*60}")
print(f" ВСЕ ЗАДАЧИ ПО ФОРМАТАМ ХРАНЕНИЯ ВЫПОЛНЕНЫ")
print(f"{'='*60}")