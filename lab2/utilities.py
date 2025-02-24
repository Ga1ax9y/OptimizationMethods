import numpy as np

def find_inverse_with_another_column(A_inverse : np.ndarray, column : np.ndarray, column_index : int):
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