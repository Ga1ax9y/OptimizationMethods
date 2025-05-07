from copy import deepcopy
import numpy as np

def transport_problem(a, b, c):
    print("Построение начального базисного плана методом северо-западного угла")

    x = np.zeros((len(a), len(b)))
    B = []
    i = j = 0
    final_i = final_j = 0
    print(len(a),len(b))
    while i < len(a) and j < len(b):
        # transfer_value - сколько ресурса выделено на текущем шаге
        transfer_value = min(a[i], b[j])
        if (i,j) not in B:
            x[i, j] = transfer_value
            B.append((i, j))
        a[i] -= transfer_value
        b[j] -= transfer_value
        if a[i] == 0:
            i += 1
            if i == len(a) and j < len(b):
                while j < len(b):
                    if (len(a) - 1, j) not in B:
                        B.append((len(a) - 1, j))
                    j += 1

        elif b[j] == 0:
            j += 1
            if j == len(b) and i < len(a):
                while i < len(a):
                    if (i, len(b) - 1) not in B:
                        B.append((i, len(b) - 1))
                    i += 1

    #B = list(dict.fromkeys(B))
    print()
    print("Первоначальный базисный план x:")
    print(x)
    print(B)
    print()
    print(f"Cтоимость: {np.trace(c @ x.T)}")
    m, n = c.shape
    iteration = 0
    while True:
        print()
        iteration += 1
        print(f"Итерация {iteration}")

        # Расчет потенциалов u,v
        u = np.full(m, None)
        v = np.full(n, None)

        first_cell = B[0]
        # Предродлжим что первое u = 0
        u[first_cell[0]] = 0.0
        stack = [first_cell]

        while stack:
            i, j = stack.pop()
            if u[i] is not None and v[j] is None:
                v[j] = c[i, j] - u[i]
                for cell in B:
                    if cell[1] == j and cell != (i, j):
                        stack.append(cell)
            elif v[j] is not None and u[i] is None:
                u[i] = c[i, j] - v[j]
                for cell in B:
                    if cell[0] == i and cell != (i, j):
                        stack.append(cell)
        print("Полученные u:", u)
        print("Полученные v:", v)
        # Ищем новую базисную клетку
        new_basis_cell = None
        for i in range(m):
            if new_basis_cell:
                break
            for j in range(n):
                if (i, j) not in B and u[i] + v[j] > c[i, j]:
                    new_basis_cell = (i, j)
                    break
        # Если не нашлось -> оптимальный план
        if new_basis_cell is None:
            print("Оптимальный план:")
            print(x)
            print()
            print(f"Финальная стоимость: {np.trace(c @ x.T)}")
            return (x, B)

        # Обновление базиса
        B.append(new_basis_cell)
        B = sorted(B)

        B_copy = deepcopy(B)
        cells_removed = True

        while cells_removed:
            cells_removed = False
            # Удаление клеток в строках с 0 и 1 базисной клеткой
            for i in range(m):
                amount = 0
                for j in range(n):
                    if (i, j) in B_copy:
                        amount += 1
                if amount < 2:
                    for j in range(n):
                        if (i, j) in B_copy:
                            B_copy.remove((i, j))
                            cells_removed = True
            # Удаление клеток в столбцах с 0 и 1 базисной клеткой
            for j in range(n):
                amount = 0
                for i in range(m):
                    if (i, j) in B_copy:
                        amount += 1
                if amount < 2:
                    for i in range(m):
                        if (i, j) in B_copy:
                            B_copy.remove((i, j))
                            cells_removed = True

        cell_marks = {item: None for item in B_copy}
        cell_marks[new_basis_cell] = True
        stack = [new_basis_cell]

        while stack:
            current_basis = stack.pop()
            mark = not cell_marks[current_basis]

            for (i, j), value in cell_marks.items():
                if value is None and (current_basis[0] == i or current_basis[1] == j):
                    cell_marks[(i, j)] = mark
                    stack.append((i, j))

        # tetta - минимальное число, которое не приведет к отрицательности
        tetta = np.inf
        for (i, j), mark in cell_marks.items():
            if not mark:
                tetta = min(tetta, x[i, j])

        for (i, j), mark in cell_marks.items():
            if mark:
                x[i, j] += tetta
            else:
                x[i, j] -= tetta

        for i, j in B:
            if x[i, j] == 0 and not cell_marks.get((i, j), True):
                B.remove((i, j))
                break

        print(f"Обновленный базисный план:")
        print(x)
        print()
        print(f"Стоимость: {np.trace(c @ x.T)}")

if __name__ == "__main__":
    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array(
        [
            [8, 4, 1],
            [8, 4, 3],
            [9, 7, 5],
        ]
    )

    print("Исходные данные: ")
    print('-' * 60, end="\n\n")
    print("Вектор предложения a:", a, sep='\n', end='\n\n')
    print("Вектор спроса b:", b, sep='\n', end='\n\n')
    print("Матрица  стоимости перевозок c:")
    print(c)
    print('-' * 60, end="\n\n")

    transport_problem(a, b, c)
