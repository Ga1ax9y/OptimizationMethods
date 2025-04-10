import numpy as np

def find_inverse_with_another_column(A_inverse, column, column_index):
    n = A_inverse.shape[0]

    l = A_inverse.dot(column)
    l_column_index = l[column_index]

    if (l_column_index == 0):
        return None

    l[column_index] = -1
    l_new = (-1 / l_column_index) * l

    Q = np.eye(n)
    Q[:, column_index] = l_new

    A_result_inverse = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if (i == column_index):
                A_result_inverse[i, k] = l_new[i] * A_inverse[i, k]
                continue

            A_result_inverse[i, k] = A_inverse[i, k] + l_new[i] * A_inverse[column_index, k]

    return A_result_inverse


class Simplex:
    def __init__(self, code, error = None, answer = None, B_set = None):
        self.code = code
        self.error = error
        self.answer = answer
        self.B_set = B_set

def dual_simplex(c, A, b, B_init):
    _, n = A.shape
    B = B_init.copy()
    A_B_inv = None
    index_replaceble = None
    column_index_replaceble = None

    while True:
        # 1 Составим базисную матрицу A_B и найдем для нее обратную матрицу A_B_inv
        A_B = A[:, B]

        if A_B_inv is None:
            try:
                A_B_inv = np.linalg.inv(A_B)
            except np.linalg.LinAlgError:
                return Simplex(1, "Задача несовместна (вырожденная матрица A_B)")
        else:
            A_B_inv = find_inverse_with_another_column(A_B_inv, A[:, column_index_replaceble], index_replaceble)

        # 2 Сформируем вектор c_B, состоящий из компонент вектора c с базисными индексами.
        c_B = c[B]

        # 3 Находим базисный допустимый план двойственной задачи y = c_B * A_B_inv.
        y = c_B.dot(A_B_inv)

        # 4 Находим псевдоплан κ = (κB,κN), соответствующий текущему базисному допустимому плану y, κ_B = A_B_inv * b, κN = 0.
        kappa_B = A_B_inv.dot(b)
        kappa = np.zeros(n)
        kappa[B] = kappa_B

        # 5 Если κ ⩾ 0, то κ — оптимальный план прямой задачи и метод завершает свою работу.
        if (kappa >= 0).all():
            return Simplex(0, "None", kappa, B)

        #? 6 Выделим отрицательную компоненту псевдоплана κ и сохраним ее индекс. Этот индекс базисный jk ∈ B.
        negative_indices = np.where(kappa[B] < 0)[0]
        k = negative_indices[0]

        # 7 Пусть ∆y — это k-я строка матрицы A_B_inv . Для каждого индексаj ∈{1,2,...,n} \ B вычислим  µj = ∆y * Aj, где Aj — это j-ый столбец матрицы A.
        delta_y = A_B_inv[k, :]
        non_B = [j for j in range(n) if j not in B]
        mu = {}
        for j in non_B:
            mu[j] = delta_y.dot(A[:, j])

        # 8 Если для каждого индекса j ∈ {1,2,...,n}\B выполняется µj ⩾ 0, то прямая задача несовместна и метод завершает свою работу.
        if all(mu_j >= 0 for mu_j in mu.values()):
            return Simplex(1, "Задача несовместна")

        """ 9 Для каждого индекса j ∈ {1,2,...,n}  B такого, что µj < 0 вычислим
            sigmaj = (cj -Aj⋅y)/µj
            10 sigma0 = min{sigmaj : j ∈ {1,2,...,n}B ∧ µj < 0}
        """
        sigma = {}
        for j in non_B:
            if mu[j] < 0:
                sigma[j] = (c[j] - A[:, j].dot(y)) / mu[j]

        j_0 = min(sigma, key=lambda x: sigma[x])

        # 11 В множестве B заменим k-ый базисный индекс на индекс j0. Переходим на Шаг 1.
        B[k] = j_0
        index_replaceble = k
        column_index_replaceble = j_0

if __name__ == "__main__":
    example_number = 0
    c = [
    np.array([-4, -3, -7, 0, 0]),
    np.array([1, 1]),
    np.array([3, 2, 0, 0])
    ]

    A = [
        np.array([
            [-2, -1, -4, 1, 0],
            [-2, -2, -2, 0, 1]
        ]),
        np.array([
            [1, 1],
            [-1, -1]
        ]),
        np.array([
            [1, 1, 1, 0],
            [1, -1, 0, -1]
        ])
    ]

    b = [
        np.array([-1, -3/2]),
        np.array([2, -3]),
        np.array([2, 4])
    ]

    B_init = [
        [3, 4],
        [0, 1],
        [2, 3]
    ]
    print("Вектор c = ", c[example_number], sep=' ', end='\n\n')
    print("Матрица A = ", A[example_number], sep=' ', end='\n\n')
    print("Вектор b = ", b[example_number], sep=' ', end='\n\n')
    print("Множество B = ", B_init[example_number], sep=' ', end='\n\n')

    print("Ответ:", end="\n\n")
    answer = dual_simplex(c[example_number], A[example_number], b[example_number], B_init[example_number])

    if answer.code == 0:
        print("Оптимальный план задачи:")
        print("𝜅 =", answer.answer, end='\n\n')
        print("Базисные индексы B =", answer.B_set, end='\n\n')
    else:
        print("Ошибка:", answer.error)
