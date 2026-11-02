def checking(A):    # метод для проверки на нулевые строки и столбцы
    n = len(A)
    for i in range(n):     # проверка строк
        row_sum = 0
        for j in range(n):
            row_sum += A[i][j]
        if row_sum == 0:
            return False

    for j in range(n):    # проверка столбцов
        col_sum = 0
        for i in range(n):
            col_sum += A[i][j]
        if col_sum == 0:
            return False

    return True


def make_diagonally_dominant(A, b):   # метод для диагонального преобразования
    n = len(A)

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

    # Проверка диагонального преобладания
    for i in range(n):
        s = 0
        for j in range(n):
            if j != i:
                s += abs(A[i][j])
        if abs(A[i][i]) < s:
            return False
    return True


def iteration_norm(A):    # высчитывание нормы матрицы
    n = len(A)
    max_norm = 0.0

    for i in range(n):
        s = 0.0
        for j in range(n):
            if j != i:
                s += abs(A[i][j])

        value = s / abs(A[i][i])

        if value > max_norm:
            max_norm = value
    return max_norm


def gauss_seidel(A, b, eps, M):    #метод Гауса-Зейделя
    n = len(A)
    x = [0.0] * n   # начальное приближение (все нули)
    k = 1

    while True:
        delta = 0        # максимальная погрешность на итерации
        errors = [0.0] * n   # вектор погрешностей текущей итерации

        for i in range(n):
            s = 0
            for j in range(i):
                s += A[i][j] * x[j]

            for j in range(i + 1, n):
                s += A[i][j] * x[j]

            x_new = (b[i] - s) / A[i][i]

            d = abs(x_new - x[i])
            errors[i] = d   # сохраняем погрешность компоненты

            if d > delta:
                delta = d

            x[i] = x_new

        if delta < eps:
            return x, k, delta, errors, True

        if k >= M:
            return x, k, delta, errors, False

        k += 1


def input_keyboard():    # ввод с клавиатуры
    n = int(input("Введите размерность (n <= 20): "))
    eps = float(input("Введите точность ε: "))
    M = int(input("Введите максимальное число итераций M: "))

    A = []
    print("Введите матрицу A:")
    for i in range(n):
        row = list(map(float, input().split()))
        A.append(row)

    print("Введите вектор b:")
    b = list(map(float, input().split()))

    return A, b, eps, M


def input_file(filename):     # загрузка из файла
    with open(filename, 'r') as f:
        n = int(f.readline())
        eps = float(f.readline())
        M = int(f.readline())

        A = []
        for _ in range(n):
            A.append(list(map(float, f.readline().split())))

        b = list(map(float, f.readline().split()))

    return A, b, eps, M


def main():
    print("Выберите способ ввода:")
    print("1 - с клавиатуры")
    print("2 - из файла")

    choice = input("Ваш выбор: ")

    if choice == "1":
        A, b, eps, M = input_keyboard()
    elif choice == "2":
        filename = input("Введите имя файла: ")
        A, b, eps, M = input_file(filename)
    else:
        print("Неверный выбор")
        return

    n = len(A)

    if n > 20:
        print("Ошибка: n должно быть <= 20")
        return

    if checking(A) is False:
        print("Матрица включает нулевую строку/столбец. Невозможно найти решения")
        return

    print("\nПопытка достичь диагонального преобладания...")

    if make_diagonally_dominant(A, b):
        print("Диагональное преобладание достигнуто.")
    else:
        print("Предупреждение: диагональное преобладание отсутствует.")
        print("Метод может сходиться медленно или расходиться.")

    norm_C = iteration_norm(A)

    print("\nНорма итерационной матрицы ||C||:", norm_C)

    if norm_C < 1:
        print("Условие сходимости выполнено (||C|| < 1).")
    else:
        print("Условие сходимости НЕ выполнено (||C|| >= 1).")

    print(A)

    x, iterations, delta, errors, converged = gauss_seidel(A, b, eps, M)

    if converged:
        print("\nРешение найдено:")
        for i in range(n):
            print(f"x{i+1} = {x[i]}")
        print("\nКоличество итераций:", iterations)
        print("\nВектор погрешностей последней итерации:")
        for i in range(n):
            print(f"|Δx{i + 1}| = {errors[i]}")
    else:
        print("\nИтерации расходятся (достигнуто M итераций).")
        print("Последнее приближение:")
        for i in range(n):
            print(f"x{i+1} = {x[i]}")
        print("\nВектор погрешностей последней итерации:")
        for i in range(n):
            print(f"|Δx{i + 1}| = {errors[i]}")


if __name__ == "__main__":
    main()
