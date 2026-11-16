import numpy as np
import timeit
import os

print("=" * 60)
print(" ЗАДАЧА 1: Числа с плавающей точкой")
print("=" * 60)

arr32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
arr64 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
s32 = arr32.sum()
s64 = arr64.sum()
print(f"\n[1.1] Сумма float32: {s32}")
print(f"      Сумма float64: {s64}")
print(f"      Точное значение: 0.6")
print(f"      Ошибка float32: {abs(s32 - 0.6):.2e}")
print(f"      Ошибка float64: {abs(s64 - 0.6):.2e}")

eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps
print(f"\n[1.2] Машинное эпсилон float32: {eps32:.2e}")
print(f"      Машинное эпсилон float64: {eps64:.2e}")

a = np.float32(1.0000001)
b = np.float32(1.0)
print(f"\n[1.3] Катастрофическая отмена:")
print(f"      a = {a}, b = {b}")
print(f"      a - b = {a - b}  (ожидалось 0.0000001)")

x = np.float32(1e8)
y = np.float32(1e8 + 1)
print(f"      (1e8+1) - 1e8 в float32 = {y - x}  (потеря точности)")


print("\n" + "=" * 60)
print(" ЗАДАЧА 2: Индексация, срезы и broadcasting")
print("=" * 60)

A = np.arange(20).reshape(4, 5)
print(f"\n[2.1] Массив A формы (4,5):\n{A}")
print(f"      Вторая строка (индекс 1): {A[1, :]}")
print(f"      Столбец с индексом 3:     {A[:, 3]}")
print(f"      Подмассив строк 1-3, столбцов 2-4:\n{A[1:4, 2:5]}")

rows_idx = [0, 2, 3]
cols_idx = [1, 4]
selected = A[np.ix_(rows_idx, cols_idx)]
print(f"\n[2.2] Fancy-индексация (строки {rows_idx}, столбцы {cols_idx}):\n{selected}")

B = np.random.randint(-10, 10, size=(4, 5))
print(f"\n[2.3] Исходный массив B:\n{B}")
B[B < 0] = 0
print(f"      После замены отрицательных на 0:\n{B}")

B_arr = np.arange(4).reshape(4, 1)
C_arr = np.arange(5).reshape(1, 5)
result = B_arr + C_arr
print(f"\n[2.4] Broadcasting: B(4,1) + C(1,5) =\n{result}")


print("\n" + "=" * 60)
print(" ЗАДАЧА 3: Memory-mapped файлы и оптимизация кэша")
print("=" * 60)

filename = "mmap_array.dat"

mmap_arr = np.memmap(filename, dtype='float64', mode='w+', shape=(1000, 1000))
mmap_arr[:] = np.random.rand(1000, 1000)
mmap_arr.flush()
print(f"\n[3.1] Создан memory-mapped файл: {filename}")
print(f"      Форма: {mmap_arr.shape}, тип: {mmap_arr.dtype}")
print(f"      Размер на диске: {os.path.getsize(filename) / 1024:.1f} KB")

mmap_read = np.memmap(filename, dtype='float64', mode='r', shape=(1000, 1000))
row_5 = mmap_read[5, :]
col_7 = mmap_read[:, 7]
print(f"\n[3.2] Строка 5 (первые 5 эл.): {row_5[:5]}")
print(f"      Столбец 7 (первые 5 эл.): {col_7[:5]}")

setup = "import numpy as np; a = np.memmap('mmap_array.dat', dtype='float64', mode='r', shape=(1000,1000))"
t_row = timeit.timeit("np.sum(a, axis=1)", setup=setup, number=100)
t_col = timeit.timeit("np.sum(a, axis=0)", setup=setup, number=100)

print(f"\n[3.3] Обход по строкам (row-major):    {t_row:.4f} сек")
print(f"      Обход по столбцам (column-major): {t_col:.4f} сек")
print(f"      Разница: в {t_col/t_row:.1f} раз")
print(f"      Причина: NumPy хранит данные в row-major (C-стиль).")
print(f"      При обходе по строкам — последовательный доступ к памяти,")
print(f"      хорошее использование кэш-линий. По столбцам — промахи кэша.")


print("\n" + "=" * 60)
print(" ЗАДАЧА 4: Разреженные матрицы")
print("=" * 60)

from scipy import sparse

np.random.seed(42)
dense_small = np.random.rand(5, 5)
dense_small[dense_small > 0.2] = 0
csr = sparse.csr_matrix(dense_small)
print(f"\n[4.1] Разреженная матрица CSR (5x5, ~20% ненулевых):\n{csr.toarray()}")
print(f"      Ненулевых элементов: {csr.nnz} из 25")

coo = csr.tocoo()
csc = csr.tocsc()
print(f"\n[4.2] COO координаты: {list(zip(coo.row.tolist(), coo.col.tolist()))}")
print(f"      COO значения:   {coo.data}")
print(f"      CSC матрица:\n{csc.toarray()}")

vec = np.random.rand(5)
result = csr.dot(vec)
print(f"\n[4.3] Умножение CSR на вектор: {result}")
print(f"      Проверка (плотная):      {dense_small.dot(vec)}")

from scipy.sparse import random as sparse_random

N = 1000
sp_mat = sparse_random(N, N, density=0.01, format='csr', dtype=np.float64)
dense_mat = sp_mat.toarray()
v = np.random.rand(N)

t_sparse = timeit.timeit("sp_mat.dot(v)", globals=globals(), number=100)
t_dense = timeit.timeit("dense_mat.dot(v)", globals=globals(), number=100)

print(f"\n[4.4] Разреженная (CSR) 1000x1000, плотность 0.01:")
print(f"      Разреженная: {t_sparse:.4f} сек")
print(f"      Плотная:     {t_dense:.4f} сек")
print(f"      Ускорение: в {t_dense/t_sparse:.1f} раз")


print("\n" + "=" * 60)
print(" ЗАДАЧА 5: Написание нативного расширения с Cython")
print("=" * 60)

def sum_sq_python(lst):
    """Вычисляет сумму квадратов элементов списка."""
    s = 0
    for x in lst:
        s += x * x
    return s

cython_code = '''
# Файл: sum_sq_cython.pyx
def sum_sq_cython(double[:] lst):
    cdef double s = 0.0
    cdef int i
    cdef int n = lst.shape[0]
    for i in range(n):
        s += lst[i] * lst[i]
    return s
'''
print(f"\n[5.2] Код Cython (файл sum_sq_cython.pyx):\n{cython_code}")

try:
    from sum_sq_cython import sum_sq_cython
    cython_available = True
    print("      Cython-модуль успешно импортирован!")
except ImportError:
    cython_available = False
    print("      Cython-модуль не найден. Запустите:")
    print("      python setup.py build_ext --inplace")

arr = np.random.rand(10**7)
lst = arr.tolist()

t_py = timeit.timeit("sum_sq_python(lst)", globals=globals(), number=3)
print(f"\n[5.3] Массив из 10^7 элементов:")
print(f"      Python: {t_py:.4f} сек")

if cython_available:
    t_cy = timeit.timeit("sum_sq_cython(arr)", globals=globals(), number=3)
    print(f"      Cython: {t_cy:.4f} сек")
    print(f"      Ускорение: в {t_py/t_cy:.1f} раз")
else:
    print(f"      Cython: не скомпилирован (см. инструкцию выше)")
    print(f"      Ожидаемое ускорение: в 10-50 раз")


print("\n" + "=" * 60)
print(" ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ")
print("=" * 60)
