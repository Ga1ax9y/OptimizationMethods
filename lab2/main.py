from utilities import np
from utilities import find_inverse_with_another_column
from data import *

class SimplexResult:
    def __init__(self, result_code, error_message = None, result = None, B_set = None):
        self.result_code = result_code
        self.error_message = error_message
        self.result = result
        self.B_set = B_set

def simplex_core_phase(c : np.ndarray, A : np.ndarray, x_init : np.ndarray, B_init : np.ndarray):
    m, _ = A.shape
    x = x_init.copy()
    B = B_init.copy()

    A_B_inv_previous = None

    while True:
        # Шаг 1: Построить базисную матрицу и обратную
        if (A_B_inv_previous is None):
            A_B = A[:, B]

            try:
                A_B_inv = np.linalg.inv(A_B)
            except np.linalg.LinAlgError:
                return SimplexResult(2, "Базисная матрица вырожденна.")

            A_B_inv_previous = A_B_inv
        else:
            A_B_inv = find_inverse_with_another_column(A_B_inv_previous, A[:, B[k]], k)

            if (A_B_inv is None):
                return SimplexResult(2, "Базисная матрица вырожденна.")

        # Шаг 2: Вектор c_B
        c_B = c[B]

        # Шаг 3: Вектор потенциалов
        u = c_B @ A_B_inv

        # Шаг 4: Вектор оценок
        delta = u @ A - c

        # Шаг 5: Проверка оптимальности
        if np.all(delta >= 0):
            return SimplexResult(0, None, x, B)

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
            return SimplexResult(1, None, "Целевой функционал задачи не ограничен сверху на множестве допустимых планов.")

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
    A_list = [
        np.array([
        [-1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
    ]),
        np.array([
        [2, 1, 1, 0, 0, 0],
        [1, 2, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1]
    ]),
        np.array([
        [2, 1, 1, 1, 0, 0, 0],
        [1, 2, 1, 0, 1, 0, 0],
        [1, 1, 2, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 1]
    ]),
        np.array([
        [1, -1, 1, 0],
        [-1, 1, 0, 1]
    ])
    ]

    c_list = [
        np.array([1, 1, 0, 0, 0]),
        np.array([4, 3, 0, 0, 0, 0]),
        np.array([5, 4, 3, 0, 0, 0, 0]),
        np.array([1, 1, 0, 0])
    ]

    B_init_list = [
        [2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [2, 3]
    ]

    x_init_list = [
        np.array([0, 0, 1, 3, 2]),
        np.array([0, 0, 4, 3, 2, 1]),
        np.array([0, 0, 0, 5, 4, 3, 2]),
        np.array([0, 0, 1, 1])
    ]

    b_list = [
        np.array([4, 3, 2, 1])
    ]
    c = c_list[0]
    A = A_list[0]
    x_init = x_init_list[0]
    B_init = B_init_list[0]

    print("Вектор c:", c, sep='\n', end='\n\n')
    print("Матрица A:", A, sep='\n', end='\n\n')
    print("Допустимый план x:", x_init, sep='\n', end='\n\n')
    print("Множество B:", B_init, sep='\n', end='\n\n')
    print("-" * 30, end="\n\n")

    result = simplex_core_phase(c, A, x_init, B_init)

    if (result.result_code == 0):
        print("Оптимальный план:\nx =", result.result)
        print("\nКонечное множество индексов:\nB =", result.B_set)
    elif (result.result_code == 1):
        print(result.result)
    elif (result.result_code == 2):
        print("Ошибка:", result.error_message, sep='\n')

if __name__ == "__main__":
    main()
