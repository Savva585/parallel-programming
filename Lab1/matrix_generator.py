import numpy as np

np.random.seed(42)

sizes = [100, 500]
for n in sizes:
    matrix = np.random.randint(0, 10, (n, n))
    filename = f"matrix_{n}.txt"
    np.savetxt(filename, matrix, fmt='%.6f', delimiter=' ') 
    print(f"Файл {filename} создан.")