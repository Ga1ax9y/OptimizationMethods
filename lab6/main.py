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
    def __init__(self, result_code, error_message=None, result=None, J_b_set=None, J_b_star_set=None):
        self.result_code = result_code
        self.error_message = error_message
        self.result = result
        self.J_b_set = J_b_set
        self.J_b_star_set = J_b_star_set

def solve_quadratic(c, D, A, initial_x, J_b, J_b_star):
    x = np.array(initial_x, dtype=float)
    n = len(x)
    m = A.shape[0]
    A_b_inv = None
    is_replace = False
    index_replaceble = None
    column_index_replaceble = None

    while True:
        # 1 находим c_x, u_x, delta_x
        c_x = c + D @ x
        A_b = A[:, J_b]

        if A_b_inv is None:
            try:
                A_b_inv = np.linalg.inv(A_b)
            except np.linalg.LinAlgError:
                return Simplex(1, "Матрица A_b является вырожденной")
        elif is_replace:
            A_b_inv = find_inverse_with_another_column(A_b_inv, A[:, column_index_replaceble], index_replaceble)
            is_replace = False

        u_x = -c_x[J_b] @ A_b_inv
        delta_x = u_x @ A + c_x

        # 2 проверка delta_x > 0 на оптимальный план
        if np.all(delta_x >= 0):
            return Simplex(0, result=x, J_b_set=J_b, J_b_star_set=J_b_star)

        # 3 выбираем индекс первой отрицательной компоненты
        j_0 = np.where(delta_x < 0)[0]
        if j_0.size > 0:
            j_0 = j_0[0]
        else:
            j_0 = None

        # 4 Рассчет l
        l = np.zeros(n)
        l[j_0] = 1
        D_star = D[np.ix_(J_b_star, J_b_star)]
        A_b_star = A[:, J_b_star]
        H_top = np.hstack((D_star, A_b_star.T))
        H_bottom = np.hstack((A_b_star, np.zeros((m, m))))
        H = np.vstack((H_top, H_bottom))
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return Simplex(1, "Матрица H является вырожденной")
        # столбец из D с индексами из B*
        b_star_top = D[:, j_0][J_b_star]

        b_star_bottom = A[:, j_0]
        b_star = np.concatenate((b_star_top, b_star_bottom))
        x_hat = -H_inv @ b_star
        l_b_star = x_hat[:len(J_b_star)]
        l[J_b_star] = l_b_star

        # 5 Рассчет дельты и тетта
        delta = l @ D @ l
        if delta <= 0:
            theta_j_0 = np.inf
        else:
            theta_j_0 = abs(delta_x[j_0]) / delta
        thetas = []
        for j in J_b_star:
            if l[j] < 0:
                thetas.append((-x[j] / l[j], j))
            else:
                thetas.append((np.inf, j))
        thetas.append((theta_j_0, j_0))
        theta_0, j_star = min(thetas, key=lambda x: x[0])
        if theta_0 == np.inf:
            return Simplex(1, "Целевая функция задачи не ограничена снизу на множестве допустимых планов")

        # 6 Обновление
        x += theta_0 * l
        # 1: j* == j0
        if j_star == j_0:
            J_b_star = np.append(J_b_star, j_star)

        # 2: j* ∈ J_b* \ J_b
        elif j_star in [j for j in J_b_star if j not in J_b]:
            J_b_star = J_b_star[J_b_star != j_star]

        # 3: j* ∈ J_b и существует j+ ∈ J_b* \ J_b с (A_b_inv @ A_j+)[s] != 0
        elif j_star in J_b:
            s = np.where(J_b == j_star)[0][0]
            found = False
            for j_plus in [j for j in J_b_star if j not in J_b]:
                A_j_plus = A[:, j_plus]
                if abs((A_b_inv @ A_j_plus)[s]) > 0:
                    J_b[s] = j_plus
                    J_b_star = J_b_star[J_b_star != j_star]
                    found = True

                    index_replaceble = s
                    column_index_replaceble = j_plus
                    is_replace = True

                    break

            # 4: j* ∈ J_b и (J_b == J_b* или все (A_b_inv @ A_j+)[s] == 0)
            if not found:
                J_b[s] = j_0
                J_b_star[np.where(J_b_star == j_star)[0][0]] = j_0

                index_replaceble = s
                column_index_replaceble = j_0
                is_replace = True

if __name__ == "__main__":
    example_number = 0
    c = [
    np.array([-8, -6, -4, -6]),
    np.array([-1, -1])
    ]
    D = [
    np.array([
    [2, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
    ]),
    np.array([[-1, -1],
            [-1, -1]])
    ]
    A = [
    np.array([
    [1, 0, 2, 1],
    [0, 1, -1, 2]
    ]),
    np.array([[1, -1]])
    ]
    initial_x = [
    np.array([2, 3, 0, 0]),
    np.array([1, 1])
    ]
    J_b = [
    np.array([0, 1]),
    np.array([0]),
    ]
    J_b_star = [
    np.array([0, 1]),
    np.array([0]),
    ]

    print("Вектор c:", c[example_number], sep='\n', end='\n\n')
    print("Матрица A:", A[example_number], sep='\n', end='\n\n')
    print("Матрица D:", D[example_number], sep='\n', end='\n\n')
    print("Изначальный вектор x:", initial_x[example_number], sep='\n', end='\n\n')
    print("Множество J_b:", J_b[example_number], sep='\n', end='\n\n')
    print("Множество J_b_star:", J_b_star[example_number], sep='\n', end='\n\n')
    print("-" * 30, end="\n\n")

    result = solve_quadratic(c[example_number], D[example_number], A[example_number], initial_x[example_number], J_b[example_number], J_b_star[example_number])

    if result.result_code == 0:
        print("Оптимальный план задачи:")
        print("x =", result.result, end='\n\n')
        print("Конечное множество J_b:", result.J_b_set, end='\n\n')
        print("Конечное множество J_b_star:", result.J_b_star_set, end='\n\n')
    else:
        print("Ошибка:", result.error_message)
