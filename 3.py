import math

# Функции
def f1(x):
    return 4*x**3 - 5*x**2 + 6*x - 7

def f2(x):
    return -2*x**3 - 4*x**2 + 8*x - 4

def f3(x):
    return math.exp(x)

# Выбор функции
def get_function(choice):
    if choice == 1:
        return f1
    elif choice == 2:
        return f2
    elif choice == 3:
        return f3


# Метод левых прямоугольников
def left_rectangles(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + i * h)
    return s * h


# Метод правых прямоугольников
def right_rectangles(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + (i + 1) * h)
    return s * h


# Метод средних прямоугольников
def middle_rectangles(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + (i + 0.5) * h)
    return s * h


# Метод трапеций
def trapezoidal(f, a, b, n):
    h = (b - a) / n
    s = (f(a) + f(b)) / 2
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


# Метод Симпсона
def simpson(f, a, b, n):
    if n % 2 != 0:
        n = n + 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        if i % 2 != 0:
            s += 4 * f(a + i * h)
        else:
            s += 2 * f(a + i * h)
    return s * h / 3


# Порядки точности методов
order = {
    left_rectangles: 1,
    right_rectangles: 1,
    middle_rectangles: 2,
    trapezoidal: 2,
    simpson: 4
}

# Правило Рунге
def runge(I_n, I_2n, method):
    p = order[method]
    return abs(I_2n - I_n) / (2**p - 1)


def integrate(method, f, a, b, eps):
    n = 4
    I_n = method(f, a, b, n)
    iteration = 1
    
    print("\n")
    print(f"{'Итерация':^10} {'n':^10} {'I_n':^20} {'I_2n':^20} {'Погрешность Рунге':^15}")
    print("_"*90)
    
    while True:
        n2 = n * 2
        I_2n = method(f, a, b, n2)
        
        err = runge(I_n, I_2n, method)
        
        print(f"{iteration:^10} {n:^10} {I_n:^20.10f} {I_2n:^20.10f} {err:^15.10f}")
        
        if err < eps:
            print("_"*90)
            print(f"Достигнута требуемая точность {eps}")
            return I_2n, n2
        else:
            n = n2
            I_n = I_2n
            iteration += 1


# Выбор метода
def get_method(choice):
    if choice == 1:
        return left_rectangles
    elif choice == 2:
        return right_rectangles
    elif choice == 3:
        return middle_rectangles
    elif choice == 4:
        return trapezoidal
    elif choice == 5:
        return simpson

        
# Основная программа
print("Выберите функцию:")
print("1: 4*x^3 - 5*x^2 + 6*x - 7")
print("2: -2*x^3 - 4*x^2 + 8*x - 4")
print("3: e^x")

f_choice = int(input())
f = get_function(f_choice)

a = float(input("Введите a: "))
b = float(input("Введите b: "))
eps = float(input("Введите точность: "))

print("Выберите метод:")
print("1: Левые прямоугольники")
print("2: Правые прямоугольники")
print("3: Средние прямоугольники")
print("4: Трапеции")
print("5: Симпсона")

m_choice = int(input())
method = get_method(m_choice)

result, n = integrate(method, f, a, b, eps)

print(f"\nЗначение интеграла = {result:.10f}")
print(f"Число разбиений n = {n}")