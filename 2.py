import math
import numpy as np
import matplotlib

matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt

# тестируемые функции
def f1(x):
    return x ** 3 + 2.28 * x ** 2 - 1.934 * x - 3.907

def f2(x):
    return math.cos(x) - x

def f3(x):
    return math.exp(x) - 3 * x

functions = {
    1: ("x^3 + 2.28x^2 - 1.934x - 3.907", f1),
    2: ("cos(x) - x", f2),
    3: ("e^x - 3x", f3)
}


def check_interval(func, a, b, eps=1e-5):
    if a >= b:
        print("Ошибка: левый конец a должен быть меньше правого b.")
        return False, 0

    # Проверка количества корней по смене знака
    x_vals = np.linspace(a, b, 200)
    sign_changes = 0
    prev_f = func(x_vals[0])
    for x in x_vals[1:]:
        f = func(x)
        if f * prev_f < 0:
            sign_changes += 1
        prev_f = f

    # Вывод количества корней
    if sign_changes == 0:
        print("На указанном интервале корней не найдено.")
        return False, 0
    elif sign_changes == 1:
        print("На указанном интервале найден 1 корень.")
    else:
        print(f"На указанном интервале обнаружено {sign_changes} корня/ей.")

    return True, sign_changes


def fprime(f, x, h):   # 1 производная
        return (f(x + h) - f(x - h)) / (2 * h)

def fprime2(f, x, h):    # 2 производная
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)


def chord_method(f, a, b, eps, max_iter=10000):
    iterations = 0
    fa = f(a)
    fb = f(b)

    # Проверка, не являются ли границы корнем
    if abs(fa) < eps:
        return a, 0
    if abs(fb) < eps:
        return b, 0

    # Вычисляем вторую производную
    h = 1e-6
    f2a = fprime2(f, a, h)
    f2b = fprime2(f, b, h)

    # Переменные для хранения текущего интервала
    current_a = a
    current_b = b

    if fa * f2a > 0:
        # Случай 1: зафиксирован конец a (левый конец неподвижен)
        x_prev = current_b  # начинаем с правого конца

        while iterations < max_iter:
            # Формула с неподвижным a
            x_next = x_prev - f(x_prev) * (x_prev - current_a) / (f(x_prev) - f(current_a))
            iterations += 1

            # Проверка сходимости
            if abs(x_next - x_prev) < eps and abs(f(x_next)) < eps:
                return x_next, iterations

            x_prev = x_next

    elif fb * f2b > 0:
        # Случай 2: зафиксирован конец b (правый конец неподвижен)
        x_prev = current_a  # начинаем с левого конца

        while iterations < max_iter:
            # Формула с неподвижным b
            x_next = x_prev - f(x_prev) * (current_b - x_prev) / (f(current_b) - f(x_prev))
            iterations += 1

            # Проверка сходимости
            if abs(x_next - x_prev) < eps and abs(f(x_next)) < eps:
                return x_next, iterations

            x_prev = x_next

    else:
        # Если ни одно условие не выполняется, используем классический метод хорд

        while iterations < max_iter:
            # Классическая формула метода хорд
            x_next = current_a - f(current_a) * (current_b - current_a) / (f(current_b) - f(current_a))
            iterations += 1
            fx = f(x_next)

            # Проверка сходимости
            if abs(fx) < eps:
                return x_next, iterations

            # Выбираем новый интервал (ту половину, где функция имеет разные знаки)
            if f(current_a) * fx < 0:
                current_b = x_next  # корень между a и x
            else:
                current_a = x_next  # корень между x и b

    return x_prev, iterations


def secant_method(f, a, b, eps, max_iter=10000):
    iterations = 0

    # Выбор начальных приближений как в методе Ньютона
    fa = f(a)
    fb = f(b)

    if abs(fa) < eps:
        return a, 0
    if abs(fb) < eps:
        return b, 0

    # Вычисляем первую и вторую производные для выбора начального приближения
    h = 1e-6
    f2a = fprime2(f, a, h)
    f2b = fprime2(f, b, h)

    if fa * f2a > 0:
        x0 = a
    elif fb * f2b > 0:
        x0 = b
    else:
        # Если ни одно условие не выполняется, берем середину
        x0 = (a + b) / 2

    # x1 выбираем рядом с x0 (x1 = x0 + ε)
    x1 = x0 + eps

    f0 = f(x0)
    f1 = f(x1)

    # Основной цикл метода секущих
    while iterations < max_iter:
        # Защита от деления на ноль
        if abs(f1 - f0) < 1e-15:
            print("Разность значений функции слишком мала")
            break

        # Формула метода секущих
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        iterations += 1

        # Проверка сходимости
        if abs(x2 - x1) < eps or abs(f2) < eps:
            return x2, iterations

        # Переход к следующей итерации
        x0, f0 = x1, f1
        x1, f1 = x2, f2

    return x1, iterations

def create_phi(f, a, b):
    h = 1e-6
    xs = [a + i*(b-a)/100 for i in range(101)]
    derivs = [fprime(f, x, h) for x in xs]

    pmin = min(derivs)
    pmax = max(derivs)

    # Максимум модуля производной
    M = max(abs(pmin), abs(pmax), 1e-12)

    # Проверка знака производной на интервале
    if pmin >= 0:
        # f'(x) > 0
        lam = -1.0 / M
    else:
        # f'(x) < 0
        lam = 1.0 / M

    print(lam)
    # Строим φ(x)
    def phi(x):
        return x + lam * f(x)

    # Вычисляет максимальное значение |φ'(x)| на интервале для заданного λ
    def max_dphi_for(l):
        print(phi(a), fprime(f, a, h)*l + 1)
        print(phi(b), fprime(phi, b, h))
        return max(abs(1 + l * d) for d in derivs)

    max_dphi = max_dphi_for(lam)

    if max_dphi >= 1:
        print(f"Внимание: достаточное условие сходимости метода простой итерации не выполняется на [{a:.3f}, {b:.3f}] (max|φ'|={max_dphi:.3f})")

    return phi, max_dphi


def simple_iteration_method(phi, f, a, b, eps, max_iter=10000):
    iterations = 0
    fa = f(a)
    fb = f(b)

    if abs(fa) < eps:
        return a, 0
    if abs(fb) < eps:
        return b, 0

    # Вычисляем первую и вторую производные для выбора начального приближения
    h = 1e-6
    f2a = fprime2(f, a, h)
    f2b = fprime2(f, b, h)

    if fa * f2a > 0:
        x0 = a
    elif fb * f2b > 0:
        x0 = b
    else:
        # Если ни одно условие не выполняется, берем середину
        x0 = (a + b) / 2

    x0 = (a + b) / 2
    for _ in range(max_iter):
        x1 = phi(x0)
        iterations += 1
        print(x1, f(x1))
        if abs(x1-x0) < eps and abs(f(x1)) < eps:
            return x1, iterations
        x0 = x1
    return None, iterations


def input_keyboard():
    a = float(input("Введите a: "))
    b = float(input("Введите b: "))
    eps = float(input("Введите точность: "))
    return a, b, eps

def input_file():
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    print(f"\nЗагружен файл: {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()
    values = [line.strip() for line in lines if line.strip()]
    if len(values) < 3:
        print("Ошибка: в файле должно быть минимум 3 числа")
        return input_keyboard()
    try:
        a = float(values[0])
        b = float(values[1])
        eps = float(values[2])
        print("\nПрочитанные данные:")
        print(f"a = {a}, b = {b}, eps = {eps}")
        return a, b, eps
    except ValueError:
        print("Ошибка: файл содержит некорректные данные")
        return input_keyboard()


def draw_graph(f, a, b):
    x = np.linspace(a, b, 400)
    y = [f(xi) for xi in x]
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("График функции")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def main():
    print("Выберите уравнение:")
    for k in functions:
        print(f"{k} - {functions[k][0]}")
    choice = int(input())
    f = functions[choice][1]

    draw_graph(f, -5, 5)

    print("1 - ввод с клавиатуры\n2 - ввод из файла")
    mode = int(input())
    a, b, eps = input_keyboard() if mode == 1 else input_file()

    ok, num_roots = check_interval(f, a, b, eps)
    if not ok:
        return

    print("Выберите метод:\n1 - метод хорд\n2 - метод секущих\n3 - метод простой итерации")
    method = int(input())

    if method == 1:
        root, it = chord_method(f, a, b, eps)
    elif method == 2:
        root, it = secant_method(f, a, b, eps)
    else:
        phi, contracted = create_phi(f, a, b)
        if not contracted:
            print("Достаточное условие для простых итераций не выполнено. Попытка метода простой итерации...")
        root, it = simple_iteration_method(phi, f, a, b, eps)

    result = f"\nКорень: {root:.10f}\nf(x): {f(root):.6f}\nИтераций: {it}\n"
    print("1 - вывод в файл\n2 - вывод на экран")
    out = int(input())
    if out == 1:
        with open("output.txt", "w") as file:
            file.write(result)
        print("Результат сохранен в output.txt")
    else:
        print(result)

if __name__ == "__main__":
    main()