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

def main():
    c = np.array([1, 1, 0, 0, 0])
    A = np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ])
    x_init = np.array([0, 0, 1, 3, 2])
    B_init = [2, 3, 4]
    print("Исходные данные")
    print("Вектор c:", c, sep='\n', end='\n')
    print("Матрица A:", A, sep='\n', end='\n')
    print("Допустимый план x:", x_init, sep='\n', end='\n')
    print("Множество B:", B_init, sep='\n', end='\n')

    answer = simplex_main(c, A, x_init, B_init)

    if (answer.code == 0):
        print("Оптимальный план:")
        print("x =", answer.result)
        print("Конечное множество индексов:")
        print("B =", answer.B_set)
    elif (answer.code == 1):
        print(answer.result)
    else:
        print("Ошибка:", answer.error_message, sep='\n')

if __name__ == "__main__":
    main()
