import numpy as np

def manual_matrix_multiplication(Q, A_inv, n):
    """
    Умножает матрицу Q на A_inv вручную за O(n^2).

    Параметры:
    Q -- матрица Q (n x n)
    A_inv -- обратная матрица A^{-1} (n x n)
    n -- размер матрицы

    Возвращает:
    A_prime_inv -- результат умножения Q на A_inv (n x n)
    """
    A_prime_inv = np.zeros((n, n))  # Инициализация результата

    for j in range(n):  # Проходим по строкам Q
        for k in range(n):  # Проходим по столбцам A_inv
            # Умножаем j-ю строку Q на k-й столбец A_inv
            A_prime_inv[j, k] = Q[j, j] * A_inv[j, k] + (Q[j, i] * A_inv[i, k] if j != i else 0)

    return A_prime_inv

def solve_matrix(A_inv, x, i):
    """
    Решает задачу обращения модифицированной матрицы A'.

    Параметры:
    A_inv -- обратная матрица A^{-1} (n x n)
    x -- вектор-столбец (n x 1), который заменяет i-й столбец
    i -- индекс столбца, который заменяется (начиная с 0)

    Возвращает:
    A_prime_inv -- обратная матрица к A' (n x n), если она существует, иначе сообщение об ошибке
    """

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
    Q = np.eye(n)  # Единичная матрица
    Q[:, i] = l_hat  # Заменяем i-й столбец на l_hat

    # Шаг 5: Умножаем Q на A_inv
    A_prime_inv = manual_matrix_multiplication(Q, A_inv, n)

    return A_prime_inv

# Пример использования
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
    x = np.array([1, 0, 1], dtype=float)

    # Индекс заменяемого столбца (начиная с 0)
    i = 2

    result = solve_matrix(A_inv, x, i)
    if isinstance(result, str):
        print(result)
    else:
        print("Обратная матрица к A':")
        print(result)
