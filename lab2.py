import pandas as pd
import numpy as np

print("=" * 60)
print(" ЗАДАЧА 1: Создание Series и DataFrame, базовый осмотр")
print("=" * 60)

s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(f"\n[1.1] Series:\n{s}")

data = {
    'name': ['Анна', 'Борис', 'Виктор', 'Галина', 'Дмитрий', 'Елена', 'Жанна', 'Иван'],
    'department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
    'salary': [80000, 55000, 95000, np.nan, 72000, 60000, np.nan, 88000],
    'bonus': [10000, np.nan, 15000, 5000, np.nan, 8000, 12000, 20000],
    'hire_date': ['2020-01-15', '2019-06-20', '2021-03-10', '2018-11-05', 
                  '2022-07-22', '2020-09-14', '2019-02-28', '2023-01-10']
}
df = pd.DataFrame(data)
print(f"\n[1.2] DataFrame:\n{df}")

print(f"\n[1.3] Первые 3 строки:\n{df.head(3)}")
print(f"\nПоследние 3 строки:\n{df.tail(3)}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nСтатистика:\n{df.describe()}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 2: Индексация, выборка, фильтрация")
print("=" * 60)

it_employees = df[df['department'] == 'IT'][['name', 'salary']]
print(f"\n[2.1] Сотрудники IT (name, salary):\n{it_employees}")

subset = df.iloc[:3, :2]
print(f"\n[2.2] iloc[:3, :2]:\n{subset}")

high_salary = df[df['salary'] > 70000]
print(f"\n[2.3] Сотрудники с зарплатой > 70000:\n{high_salary}")

df['total_comp'] = df['salary'] + df['bonus'].fillna(0)
print(f"\n[2.4] DataFrame с total_comp:\n{df[['name', 'salary', 'bonus', 'total_comp']]}")

df = df.drop(columns=['hire_date'])
print(f"\n[2.5] DataFrame после удаления hire_date:\n{df}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 3: Обработка пропущенных значений")
print("=" * 60)

print(f"\n[3.1] Пропуски в каждом столбце:\n{df.isnull().sum()}")

df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.mean())
)
print(f"\n[3.2] После заполнения salary средним по отделу:\n{df[['name', 'department', 'salary']]}")

df['bonus'] = df['bonus'].fillna(0)
print(f"\n[3.3] После заполнения bonus нулём:\n{df[['name', 'bonus']]}")

df = df.dropna(subset=['salary'])
print(f"\n[3.4] После удаления строк с пропущенным salary:\n{df}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 4: Группировка и агрегация")
print("=" * 60)

avg_salary = df.groupby('department')['salary'].mean()
print(f"\n[4.1] Средняя зарплата по отделам:\n{avg_salary}")

dept_stats = df.groupby('department').agg({
    'salary': 'mean',
    'bonus': 'max',
    'name': 'count'
}).rename(columns={'name': 'employee_count'})
print(f"\n[4.2] Статистика по отделам:\n{dept_stats}")

pivot = df.pivot_table(index='department', values='salary', aggfunc='mean')
print(f"\n[4.3] Сводная таблица:\n{pivot}")

sorted_depts = avg_salary.sort_values(ascending=False)
print(f"\n[4.4] Отделы по убыванию средней зарплаты:\n{sorted_depts}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 5: Объединение данных")
print("=" * 60)

managers = pd.DataFrame({
    'department': ['IT', 'HR', 'Sales'],
    'manager': ['Петров', 'Сидорова', 'Козлов']
})
print(f"\n[5.1] DataFrame managers:\n{managers}")
df_merged = df.merge(managers, on='department', how='left')
print(f"\n[5.2] После merge (left join):\n{df_merged}")

new_row = pd.DataFrame([{
    'name': 'Новиков',
    'department': 'IT',
    'salary': 90000,
    'bonus': 18000,
    'total_comp': 108000,
    'manager': 'Петров'
}])
df_merged = pd.concat([df_merged, new_row], ignore_index=False)
print(f"\n[5.3] После добавления новой строки через concat:\n{df_merged}")

df_merged = df_merged.reset_index(drop=True)
print(f"\n[5.4] После сброса индекса:\n{df_merged}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 6: Комплексный мини-анализ")
print("=" * 60)

top3 = df_merged.nlargest(3, 'total_comp')[['name', 'department', 'total_comp']]
print(f"\n[6.1] Топ-3 сотрудника по total_comp:\n{top3}")

dept_summary = df_merged.groupby('department').agg({
    'salary': 'mean',
    'bonus': 'mean'
}).round(2)
print(f"\n[6.2] Средняя зарплата и бонус по отделам:\n{dept_summary}")

high_salary_share = df_merged.groupby('department').apply(
    lambda x: (x['salary'] > 75000).mean() * 100
).round(2)
print(f"\n[6.3] Доля сотрудников с зарплатой > 75000 (%):\n{high_salary_share}")


print("\n" + "=" * 60)
print(" ВСЕ ЗАДАЧИ ПО PANDAS ВЫПОЛНЕНЫ")
print("=" * 60)
