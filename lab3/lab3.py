import numpy as np

def find_inverse(A_inv, column , index):
    n = A_inv.shape[0]

    l = A_inv.dot(column)
    l_index = l[index]

    if (l_index == 0):
        return None

    l[index] = -1
    l_new = (-1 / l_index) * l

    Q = np.eye(n)
    Q[:, index] = l_new

    A_result_inv = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if (i == index):
                A_result_inv[i, k] = l_new[i] * A_inv[i, k]
                continue

            A_result_inv[i, k] = A_inv[i, k] + l_new[i] * A_inv[index, k]

    return A_result_inv

class Simplex:
    def __init__(self, code, error_message = None, result = None, B_set = None):
        self.code = code
        self.error_message = error_message
        self.result = result
        self.B_set = B_set

def simplex_main(c, A , x_init , B_init ):
    m, _ = A.shape
    x = x_init.copy()
    B = B_init.copy()

    A_B_inv_prev = None

    while True:
        # Шаг 1: Построить базисную матрицу и обратную
        if (A_B_inv_prev is None):
            A_B = A[:, B]

            try:
                A_B_inv = np.linalg.inv(A_B)
            except np.linalg.LinAlgError:
                return Simplex(2, "Базисная матрица вырожденна.")

            A_B_inv_prev = A_B_inv
        else:
            A_B_inv = find_inverse(A_B_inv_prev, A[:, B[k]], k)

            if (A_B_inv is None):
                return Simplex(2, "Базисная матрица вырожденна.")

        # Шаг 2: Вектор c_B
        c_B = c[B]

        # Шаг 3: Вектор потенциалов
        u = c_B @ A_B_inv

        # Шаг 4: Вектор оценок
        delta = u @ A - c

        # Шаг 5: Проверка оптимальности
        if np.all(delta >= 0):
            return Simplex(0, None, x, B)

        # Шаг 6: Первая отрицательная компонента
        j_0 = np.where(delta < 0)[0][0]

        # Шаг 7: Вычисление вектора z
        A_j_0 = A[:, j_0]
        z = A_B_inv @ A_j_0

        # Шаг 8: Вычисление theta
        theta = np.zeros(m)
        for i in range(m):
            z_i = z[i]
            if z_i <= 0:
                theta[i] = np.inf
            else:
                x_j_i = x[B[i]]
                theta[i] = x_j_i / z_i

        # Шаг 9: Минимальное theta
        theta_0 = np.min(theta)

        # Шаг 10: Проверка условия неограниченности
        if theta_0 == np.inf:
            return Simplex(1, None, "Целевой функционал задачи не ограничен сверху на множестве допустимых планов.")

        # Шаг 11: Индекс k для замены
        k = np.argmin(theta)
        j_star = B[k]

        # Шаг 12: Обновление базиса
        B[k] = j_0

        # Шаг 13: Обновление плана x
        x_new = x.copy()
        x_new[j_0] = theta_0
        for i in range(m):
            if i != k:
                x_new[B[i]] = x[B[i]] - theta_0 * z[i]
        x_new[j_star] = 0
        x = x_new

# Лабораторная 3
def start_phase_simplex(A, b):
    # Шаг 1: Преобразование b и A для b>=0
    m, n = A.shape
    b = b.copy().astype(float)
    A = A.copy().astype(float)

    for i in range(m):
        if b[i] < 0:
            b[i] *= -1
            A[i, :] *= -1

    # Шаг 2-3: Создание вспомогательной задачи
    c_wave = np.zeros(n + m)
    c_wave[n:] = -1

    A_wave = np.hstack((A, np.eye(m)))
    x_init = np.zeros(n + m)
    x_init[n:] = b

    B_init = np.arange(n, n + m)

    # Шаг 4: Решение вспомогательной задачи
    result = simplex_main(c_wave, A_wave, x_init, B_init)

    if result.code != 0:
        print(result.error_message)
        return Simplex(result.code, "Вспомогательная задача не была решена")

    # Шаг 5: Проверка искусственных переменных
    artificial_vars = result.result[n:]
    if not np.allclose(artificial_vars, 0):
        return Simplex(3, "Задача несовместна")

    # Шаг 6: Формирование x для исходной задачи
    x_original = result.result[:n]
    B = list(result.B_set)

    # Шаги 7-9: Корректировка базиса
    while True:
        # Поиск искусственных переменных в базисе
        artificial_in_B = [j for j in B if j >= n]
        if not artificial_in_B:
            break

        # Выбор максимального искусственного индекса
        j_k = max(artificial_in_B)
        k = B.index(j_k)
        i = j_k - n

        # Проверка исходных переменных вне базиса
        non_basis = [j for j in range(n) if j not in B]
        found = False

        # Вычисление A_B_inv
        A_B = A_wave[:, B]
        try:
            A_B_inv = np.linalg.inv(A_B)
        except np.linalg.LinAlgError:
            return Simplex(2, "Вырожденная базисная матрица при преобразовании")

        for j in non_basis:
            A_j = A_wave[:, j]
            l_j = A_B_inv @ A_j
            if not np.isclose(l_j[k], 0):
                B[k] = j
                found = True
                break

        if not found:
            # удаление
            A = np.delete(A, i, axis=0)
            b = np.delete(b, i, axis=0)

            print("Задача: ", A," = ", b)
            B.remove(j_k)
            A_wave = np.delete(A_wave, i, axis=0)

    if any(j >= n for j in B):
        return Simplex(5, "Невозможно удалить все искусственные переменные")

    return Simplex(0, None, x_original, B)

# из лабораторной работы
if __name__ == "__main__":
    c = np.array([1, 0, 0])
    A = np.array([[1, 1, 1], [2, 2, 2]])
    b = np.array([0,-13])

    result = start_phase_simplex(A, b)
    if result.code == 0:
        print("Базисный допустимый план:","x =", result.result)
        print("Базисные индексы B =", result.B_set)
    else:
        print("Ошибка:", result.error_message)
