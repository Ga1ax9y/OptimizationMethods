import numpy as np

def matrix_multiplication(Q, A_inv, n):
    A_prime_inv = np.zeros((n, n))

    for j in range(n):
        for k in range(n):
            A_prime_inv[j, k] = Q[j, j] * A_inv[j, k] + (Q[j, i] * A_inv[i, k] if j != i else 0)

    return A_prime_inv

def solve_matrix(A_inv, x, i):
    # Шаг 1: Вычисляем l = A^{-1} * x
    l = A_inv @ x

    # Проверяем, равен ли l[i] нулю
    if l[i] == 0:
        return "Матрица A' необратима"

    # Шаг 2: Формируем вектор l_wave, заменяя i-й элемент на -1
    l_wave = l.copy()
    l_wave[i] = -1

    # Шаг 3: Вычисляем l_hat = (-1 / l[i]) * l_wave
    l_hat = (-1 / l[i]) * l_wave


    # Шаг 4: Формируем матрицу Q, заменяя i-й столбец единичной матрицы на l_hat
    n = A_inv.shape[0]
    Q = np.eye(n)
    Q[:, i] = l_hat

    # Шаг 5: Умножаем Q на A_inv
    A_prime_inv = matrix_multiplication(Q, A_inv, n)

    return A_prime_inv

if __name__ == '__main__':

    # Матрица A
    A = np.array([[1, -1, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    # Обратная матрица A^{-1}
    A_inv = np.array([[1, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=float)
    # Вектор-столбец x размера n
    x = np.array([0, 0, 0], dtype=float)

    # Индекс заменяемого столбца
    i = 2
    result = solve_matrix(A_inv, x, i)
    if isinstance(result, str):
        print(result)
    else:
        print("Обратная матрица к A':")
        print(result)
