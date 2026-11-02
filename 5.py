import numpy as np
import matplotlib.pyplot as plt
import os
import math


# Вспомогательные функции

# Проверка, являются ли узлы равноотстоящими
def is_uniform(x, tol=1e-10):
    if len(x) < 2:
        return False
    h = x[1] - x[0]   # вычисление шага h
    for i in range(2, len(x)):
        if abs((x[i] - x[i-1]) - h) > tol:    #совпадает ли с шагом h
            return False
    return True


# Вычисление таблицы разделённых разностей (треугольная матрица) для построения интерполяционного многочлена Ньютона
def divided_differences(x, y):
    n = len(x)
    dd = [[0.0]*n for _ in range(n)]
    for i in range(n):
        dd[i][0] = y[i]     # нулевой столбец заполняем значениями y[i]
    for j in range(1, n):
        for i in range(n - j):
            dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (x[i+j] - x[i])    # формула разделённой разности порядка j
    return dd


# Вычисление таблицы конечных разностей вперёд
def finite_differences(y):
    # для метода Гаусса
    n = len(y)
    fd = [[0.0]*n for _ in range(n)]
    for i in range(n):
        fd[i][0] = y[i]      # нулевой столбец = значения y[i]
    for j in range(1, n): 
        for i in range(n - j):
            fd[i][j] = fd[i+1][j-1] - fd[i][j-1]    # конечная разность k-го порядка
    return fd


def print_divided_table(x, dd):
    n = len(x)
    print("\nТаблица разделённых разностей:")
    print("x\t\ty\t\tf(,)\t\tf(,,)\t\tf(,,,)\t\t...")
    for i in range(n):
        row = [f"{x[i]:8.5f}", f"{dd[i][0]:10.6f}"]
        for j in range(1, n - i):
            row.append(f"{dd[i][j]:12.6f}")
        print("\t".join(row))

def print_finite_table(x, y, fd):
    n = len(x)
    print("\nТаблица конечных разностей:")
    print("i\tx\t\ty\t\tΔ\t\tΔ²\t\tΔ³\t\t...")
    for i in range(n):
        row = [f"{i}", f"{x[i]:8.5f}", f"{y[i]:10.6f}"]
        for j in range(1, n - i):
            row.append(f"{fd[i][j]:12.6f}")
        print("\t".join(row))


# Интерполяционные методы

# Интерполяционный многочлен Лагранжа
def lagrange(x, y, xi):
    n = len(x)
    res = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (xi - x[j]) / (x[i] - x[j])    # умножаем на все дроби
        res += term    # суммируем y_i * l_i(x_i) по всем i
        # print(f"P_{i} = {res:.10f}  (добавлено {term:+.10f})")
    return res


# Первая формула Ньютона с разделёнными разностями (интерполяция вперёд)
def newton_forward_divided(x, y, xi):
    dd = divided_differences(x, y)
    n = len(x)
    res = dd[0][0]   # y0
    prod = 1.0
    # начинаем с dd[0][0] = f(x_0)
    for k in range(1, n):
        prod *= (xi - x[k-1])   # умножаем накопительное произведение
        term = dd[0][k] * prod   # добавляем dd[0][k] = Δ^k y0​ * это произведение
        res += term
        #print(f"P_{k} = {res:.10f}  (добавлено {term:+.10f})")
    return res


# Вторая формула Ньютона с разделёнными разностями (интерполяция назад)
def newton_backward_divided(x, y, xi):
    dd = divided_differences(x, y)
    n = len(x)
    # используем нижнюю диагональ: f(xn), f(xn-1,xn), f(xn-2,xn-1,xn), ...
    # в нашей таблице dd эти значения находятся на последней строке каждого столбца
    # начинаем с dd[n-1][0] = f(x_n)
    res = dd[n-1][0]
    prod = 1.0
    for k in range(1, n):
        prod *= (xi - x[n-k])
        term = dd[n-1-k][k] * prod
        res += term
        #print(f"P_{k} = {res:.10f}  (добавлено {term:+.10f})")
    return res


# Первая формула Гаусса (интерполяция вперед от центра). Для нечетного кол-ва узлов
def gauss_forward(x, y, xi):

    if not is_uniform(x):
        raise ValueError("Метод Гаусса требует равноотстоящих узлов.")
    n = len(x)
    if n % 2 == 0:
        raise ValueError("Метод Гаусса требует нечётного числа узлов.")

    h = x[1] - x[0]
    center = n // 2
    t = (xi - x[center]) / h
    fd = finite_differences(y)
    result = y[center]
    #print(result)
    product = 1.0   # накапливает произведение

    for k in range(1, n):
        if k == 1:
            index = center
            product *= t

        elif k % 2 == 0:     # для чётных k
            m = k // 2
            index = center - m
            product *= t - m + 1

        else:                   # для нечётных k
            m = (k - 1) // 2
            index = center - m
            product *= t + m

        result += product * fd[index][k] / math.factorial(k)
        # print(f"{fd[index][k]}")

    return result


# Вторая формула Гаусса (интерполяция назад от центра)
def gauss_backward(x, y, xi):

    if not is_uniform(x):
        raise ValueError("Метод Гаусса требует равноотстоящих узлов.")
    n = len(x)
    if n % 2 == 0:
        raise ValueError("Метод Гаусса требует нечётного числа узлов.")

    h = x[1] - x[0]
    center = n // 2
    t = (xi - x[center]) / h
    fd = finite_differences(y)
    
    result = y[center]
    #print(result)
    product = 1.0

    for k in range(1, n):
        if k == 1:
            index = center - 1
            product *= t

        elif k % 2 == 0:
            m = k // 2
            index = center - m
            product *= t + m - 1

        else:
            m = (k - 1) // 2
            index = center - m - 1
            product *= t - m

        result += product * fd[index][k] / math.factorial(k)
        # print(f"{fd[index][k]}")

    return result


# Ввод данных
def read_from_keyboard():
    n = int(input("Введите количество точек: "))
    x, y = [], []
    print("Введите пары x y (через пробел):")
    for _ in range(n):
        xi, yi = map(float, input().split())
        x.append(xi)
        y.append(yi)
    return x, y

def read_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    x, y = [], []
    for i in range(1, n+1):
        xi, yi = map(float, lines[i].split())
        x.append(xi)
        y.append(yi)
    return x, y

def generate_from_function():
    print("Выберите функцию:")
    print("1. sin(x)")
    print("2. cos(x)")
    print("3. exp(x)")
    print("4. x^2")
    choice = int(input("Ваш выбор: "))
    a = float(input("Левая граница интервала: "))
    b = float(input("Правая граница интервала: "))
    n = int(input("Количество узлов: "))
    x = np.linspace(a, b, n).tolist()
    if choice == 1:
        y = np.sin(x).tolist()
        func = np.sin
    elif choice == 2:
        y = np.cos(x).tolist()
        func = np.cos
    elif choice == 3:
        y = np.exp(x).tolist()
        func = np.exp
    else:
        y = (np.array(x)**2).tolist()
        func = lambda x: x**2
    return x, y, func

# Графики
def plot_results(x, y, func, xi, results):
    x_fine = np.linspace(min(x), max(x), 1000)
    methods = {

        'Лагранж':
            lambda xx: lagrange(x, y, xx),

        'Ньютон':
            lambda xx:
                newton_forward_divided(x, y, xx)
                if abs(xx - x[0]) <= abs(xx - x[-1])
                else newton_backward_divided(x, y, xx),

        'Гаусс':
            lambda xx:
                gauss_forward(x, y, xx)
                if xx >= x[len(x)//2]
                else gauss_backward(x, y, xx),
    }
    valid_methods = []
    for name, method in methods.items():

        if name not in results:
            continue

        if not isinstance(results[name], float):
            continue

        valid_methods.append((name, method))

    if len(valid_methods) == 0:
        print("Нет методов для построения графиков.")
        return
    cols = 2
    rows = math.ceil(len(valid_methods) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 5 * rows)
    )

    if len(valid_methods) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for ax, (name, method) in zip(axes, valid_methods):
        try:
            y_curve = [method(xx) for xx in x_fine]
            if func is not None:

                y_true = func(x_fine)

                ax.plot(
                    x_fine,
                    y_true,
                    'b-',
                    linewidth=2,
                    label='Исходная функция'
                )
            ax.plot(
                x_fine,
                y_curve,
                'g--',
                linewidth=2,
                label=name
            )
            ax.scatter(
                x,
                y,
                color='red',
                s=50,
                zorder=5,
                label='Узлы'
            )
            yi = method(xi)

            ax.scatter(
                [xi],
                [yi],
                color='purple',
                s=80,
                zorder=10,
                label='Точка интерполяции'
            )
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.legend()

        except Exception as e:

            ax.set_title(f"{name}\nОшибка")

            ax.text(
                0.5,
                0.5,
                str(e),
                ha='center',
                va='center'
            )

    for i in range(len(valid_methods), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Основная программа
def main():
    print(" ")
    print("Интерполяция функций")
    print("\nВыберите способ ввода данных:")
    print("1. Ввод с клавиатуры")
    print("2. Чтение из файла")
    print("3. Генерация по аналитической функции")
    mode = int(input("Ваш выбор: "))

    if mode == 1:
        x, y = read_from_keyboard()
        func = None
    elif mode == 2:
        # поиск .txt файлов в текущей папке
        files = [f for f in os.listdir('.') if f.endswith('.txt')]
        if not files:
            print("Нет файлов .txt. Создайте их вручную (например, test1.txt).")
            return
        print("\nДоступные файлы:")
        for i, fname in enumerate(files):
            print(f"{i+1}. {fname}")
        idx = int(input("Выберите номер файла: ")) - 1
        x, y = read_from_file(files[idx])
        func = None
    else:
        x, y, func = generate_from_function()

    # Сортировка по x
    sorted_points = sorted(zip(x, y))
    x, y = zip(*sorted_points)
    x = list(x)
    y = list(y)

    # Проверка корректности
    if len(set(x)) != len(x):
        print("Ошибка: узлы должны быть различными!")
        return

    # Определяем тип сетки и выводим соответствующую таблицу разностей
    uniform = is_uniform(x)
    if uniform:
        fd = finite_differences(y)
        print_finite_table(x, y, fd)
    else:
        dd = divided_differences(x, y)
        print_divided_table(x, dd)

    # Ввод аргумента
    xi = float(input("\nВведите значение аргумента x для интерполяции: "))

    results = {}

    # Лагранж
    try:
        results['Лагранж'] = lagrange(x, y, xi)
    except Exception as e:
        results['Лагранж'] = f"Ошибка: {e}"

    # Ньютон (выбор направления по расстоянию до концов)
    try:
        left_dist = abs(xi - x[0])
        right_dist = abs(xi - x[-1])
        #if left_dist <= right_dist:
        results['Ньютон'] = (newton_forward_divided(x, y, xi) + newton_backward_divided(x, y, xi))/2
        print(f"\nИспользуется первая формула Ньютона (интерполяция вперед) (от x = {x[0]})")
        print(newton_forward_divided(x, y, xi))
        #else:
        print(f"\nИспользуется вторая формула Ньютона (интерполяция назад) (от x = {x[-1]})")
        print(newton_backward_divided(x, y, xi))
    except Exception as e:
        results['Ньютон'] = f"Ошибка: {e}"

    # Гаусс (требует равномерную сетку и нечётное число узлов)
    if uniform:
        try:
            if len(x) % 2 == 1:
                center = len(x) // 2
                if xi >= x[center]:
                    results['Гаусс'] = gauss_forward(x, y, xi)
                    print(f"Используется первая формула Гаусса (интерполяция вперед от центра) (x = {x[center]})")
                else:
                    results['Гаусс'] = gauss_backward(x, y, xi)
                    print(f"Используется вторая формула Гаусса (интерполяция назад от центра) (x = {x[center]})")
            else:
                results['Гаусс'] = "Неприменим: требуется нечётное число узлов"
        except Exception as e:
            results['Гаусс'] = f"Ошибка: {e}"
    else:
        results['Гаусс'] = "Неприменим (требуется равномерная сетка)"

    print("\nРезультаты интерполяции:")
    for name, val in results.items():
        if isinstance(val, float):
            print(f"{name:20s} : {val:.20f}")
        else:
            print(f"{name:20s} : {val}")

    plot_results(x, y, func, xi, results)

    print("\nПрограмма завершена.")

if __name__ == "__main__":
    main()