import numpy as np
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def read_matrix(filename):

    with open(filename, 'r', encoding='utf-8-sig') as f:
        n = None
        for line in f:
            line = line.strip()
            if line:
                try:
                    n = int(line)
                except ValueError:
                    raise ValueError(f"Первая непустая строка должна быть целым числом (размер матрицы), получено: {line}")
                break
        if n is None:
            raise ValueError(f"Файл {filename} пуст или не содержит размера матрицы")

        mat = []
        rows_read = 0
        while rows_read < n:
            line = f.readline()
            if not line:  
                raise ValueError(f"Файл {filename}: ожидалось {n} строк, получено {rows_read}")
            line = line.strip()
            if not line:  
                continue
            parts = line.split()
            if len(parts) != n:
                raise ValueError(f"Строка {rows_read+1} в {filename} содержит {len(parts)} элементов, ожидалось {n}")
            try:
                row = [float(x) for x in parts]
            except ValueError as e:
                raise ValueError(f"Некорректное число в строке {rows_read+1} файла {filename}: {e}")
            mat.append(row)
            rows_read += 1
        extra = f.readline().strip()
        if extra:
            print(f"Предупреждение: в файле {filename} есть данные после матрицы", file=sys.stderr)

    return np.array(mat)

def main():
    if len(sys.argv) < 4:
        print("Использование: python verify.py matrixA.txt matrixB.txt result.txt")
        sys.exit(1)

    fileA, fileB, fileRes = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        A = read_matrix(fileA)
        B = read_matrix(fileB)
        C_cpp = read_matrix(fileRes)

        C_ref = np.dot(A, B)

        if np.allclose(C_cpp, C_ref, atol=1e-5):
            print("Успех. Результаты совпадают с точностью до 1e-5.")
        else:
            print("Неудача. Результаты различаются.")
            diff = np.abs(C_cpp - C_ref)
            print(f"Максимальное расхождение: {np.max(diff)}")
    except Exception as e:
        print(f"Ошибка верификации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()