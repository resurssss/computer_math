import numpy as np
import matplotlib.pyplot as plt
import warnings


# Вычисление меры отклонения S, СКО, коэффициент детерминации R^2
def compute_statistics(y_true, y_pred):
    n = len(y_true)
    S = np.sum((y_true - y_pred) ** 2)         # сумма квадратов ошибок

    if n > 0:
        delta = np.sqrt(S / n)          # средне-квадратичное отклонение
    else:
        delta = 0.0   

    y_mean = np.mean(y_true)                   
    S_tot = np.sum((y_true - y_mean) ** 2)
    if S_tot != 0:     
        R2 = 1 - (S / S_tot)        # достоверность аппроксимации (коэффициент детерминации) R^2
    else:
        R2 = 0.0   
    return S, delta, R2


# Коэффициент корреляции Пирсона для линейной связи (степень связи между x и y)
def pearson_correlation(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    x_mean = np.mean(x)     # среднее значение x
    y_mean = np.mean(y)     # среднее значение y
    verh = np.sum((x - x_mean) * (y - y_mean))     # числитель
    niz = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))          # знаменатель

    if niz == 0:                # деление на ноль
        return 0.0
    return verh / niz


# Функция для красивого табличного вывода (как на фото)
def print_table(x, y, y_pred, errors, model_name):
    n = len(x)
    print(f"\nТаблица для модели: {model_name}")
    print(f"{'№':^4} | {'X':^10} | {'Y':^10} | {'φ(x)':^12} | {'ε':^12}")
    print("-" * 60)
    
    for i in range(n):
        print(f"{i+1:^4} | {x[i]:^10.4f} | {y[i]:^10.4f} | {y_pred[i]:^12.4f} | {errors[i]:^12.4f}")
    
    print("-" * 60)


# Функция для сохранения таблицы в файл
def save_table_to_file(f, x, y, y_pred, errors, model_name):
    n = len(x)
    f.write(f"\nТаблица для модели: {model_name}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'№':^4} | {'X':^10} | {'Y':^10} | {'φ(x)':^12} | {'ε':^12}\n")
    f.write("-" * 60 + "\n")
    
    for i in range(n):
        f.write(f"{i+1:^4} | {x[i]:^10.4f} | {y[i]:^10.4f} | {y_pred[i]:^12.4f} | {errors[i]:^12.4f}\n")
    
    f.write("-" * 60 + "\n")


# Линейная модель:  y = a*x + b
def linear_approximation(x, y):
    n = len(x)
    if n < 2:
        raise ValueError("Для линейной аппроксимации нужно минимум 2 точки")
    SX = np.sum(x)
    SY = np.sum(y)
    SXX = np.sum(x * x)
    SXY = np.sum(x * y)
    
    # Решение системы методом Крамера
    det = SXX * n - SX * SX
    if abs(det) < 1e-12:
        raise ValueError("Система линейных уравнений вырождена")
    a = (SXY * n - SX * SY) / det
    b = (SXX * SY - SX * SXY) / det
    
    y_pred = a * x + b
    errors = y_pred - y
    S, delta, R2 = compute_statistics(y, y_pred)
    r = pearson_correlation(x, y)
    return a, b, S, delta, R2, r, y_pred, errors


# Квадратичная модель: y = a0 + a1*x + a2*x^2
def quadratic_approximation(x, y):
    n = len(x)
    if n < 3:
        raise ValueError("Для квадратичной аппроксимации нужно минимум 3 точки")
    
    # Вычисляем суммы
    SX = np.sum(x)
    SX2 = np.sum(x**2)
    SX3 = np.sum(x**3)
    SX4 = np.sum(x**4)
    SY = np.sum(y)
    SXY = np.sum(x * y)
    SX2Y = np.sum((x**2) * y)
    
    # Главный определитель Δ
    det = (n * (SX2 * SX4 - SX3 * SX3) -
           SX * (SX * SX4 - SX3 * SX2) +
           SX2 * (SX * SX3 - SX2 * SX2))
    
    # Проверка на вырожденность
    if abs(det) < 1e-12:
        raise ValueError("Матрица системы вырождена (det = 0)")
    
    # Определитель для a0 (заменяем первый столбец на B)
    det_a0 = (SY * (SX2 * SX4 - SX3 * SX3) -
              SX * (SXY * SX4 - SX3 * SX2Y) +
              SX2 * (SXY * SX3 - SX2 * SX2Y))
    
    # Определитель для a1 (заменяем второй столбец на B)
    det_a1 = (n * (SXY * SX4 - SX3 * SX2Y) -
              SY * (SX * SX4 - SX3 * SX2) +
              SX2 * (SX * SX2Y - SXY * SX2))
    
    # Определитель для a2 (заменяем третий столбец на B)
    det_a2 = (n * (SX2 * SX2Y - SXY * SX3) -
              SX * (SX * SX2Y - SXY * SX2) +
              SY * (SX * SX3 - SX2 * SX2))
    
    # Находим коэффициенты
    a0 = det_a0 / det
    a1 = det_a1 / det
    a2 = det_a2 / det
    
    y_pred = a0 + a1 * x + a2 * x**2
    errors = y_pred - y
    S, delta, R2 = compute_statistics(y, y_pred)
    
    return (a0, a1, a2, S, delta, R2, y_pred, errors)


# Экспоненциальная модель: y = a * exp(b*x)
def exponential_approximation(x, y):
    if np.any(y <= 0):
        raise ValueError("Для экспоненциальной модели все y должны быть > 0")
    ln_y = np.log(y)

    n = len(x)
    SX = np.sum(x)
    SY = np.sum(ln_y)
    SXX = np.sum(x*x)
    SXY = np.sum(x * ln_y)
    det = SXX * n - SX * SX
    if abs(det) < 1e-12:
        raise ValueError("Вырожденная система для экспоненциальной модели")
    b = (SXY * n - SX * SY) / det
    ln_a = (SXX * SY - SX * SXY) / det
    a = np.exp(ln_a)
    y_pred = a * np.exp(b * x)
    errors = y_pred - y
    S, delta, R2 = compute_statistics(y, y_pred)
    return a, b, S, delta, R2, y_pred, errors


# Логарифмическая модель: y = a * ln(x) + b
def logarithmic_approximation(x, y):
    if np.any(x <= 0):
        raise ValueError("Для логарифмической модели все x должны быть > 0")
    ln_x = np.log(x)
    n = len(x)
    SX = np.sum(ln_x)
    SY = np.sum(y)
    SXX = np.sum(ln_x * ln_x)
    SXY = np.sum(ln_x * y)
    det = SXX * n - SX * SX
    if abs(det) < 1e-12:
        raise ValueError("Вырожденная система для логарифмической модели")
    a = (SXY * n - SX * SY) / det
    b = (SXX * SY - SX * SXY) / det
    y_pred = a * ln_x + b
    errors = y_pred - y
    S, delta, R2 = compute_statistics(y, y_pred)
    return a, b, S, delta, R2, y_pred, errors


# Степенная модель: y = a * x^b
def power_approximation(x, y):
    if np.any(x <= 0):
        raise ValueError("Для степенной модели все x должны быть > 0")
    if np.any(y <= 0):
        raise ValueError("Для степенной модели все y должны быть > 0")
    ln_x = np.log(x)
    ln_y = np.log(y)
    n = len(x)
    SX = np.sum(ln_x)
    SY = np.sum(ln_y)
    SXX = np.sum(ln_x * ln_x)
    SXY = np.sum(ln_x * ln_y)
    det = SXX * n - SX * SX
    if abs(det) < 1e-12:
        raise ValueError("Вырожденная система для степенной модели")
    b = (SXY * n - SX * SY) / det
    ln_a = (SXX * SY - SX * SXY) / det
    a = np.exp(ln_a)
    y_pred = a * (x ** b)
    errors = y_pred - y
    S, delta, R2 = compute_statistics(y, y_pred)
    return a, b, S, delta, R2, y_pred, errors


# Функция выбора наилучшей модели (возвращает список имён с минимальным СКО)
def select_best_model(results):
    if not results:
        return [], float('inf')
    
    min_delta = float('inf')
    for name, delta, *_ in results:
        if delta < min_delta:
            min_delta = delta
    
    best_names = []
    for name, delta, *_ in results:
        if abs(delta - min_delta) < 1e-12:
            best_names.append(name)
    
    return best_names, min_delta


# Ввод данных из файла
def read_data_from_file(filename):
    try:
        data = np.loadtxt(filename, delimiter=None, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Файл должен содержать как минимум два столбца: x и y")
        x = data[:, 0]
        y = data[:, 1]
        if len(x) < 8 or len(x) > 12:
            warnings.warn(f"Предупреждение: количество точек {len(x)} не в диапазоне [8,12] (рекомендовано)")
        return x, y
    except Exception as e:
        raise ValueError(f"Ошибка чтения файла: {e}")


# Ввод данных с консоли
def read_data_from_console():
    n = int(input("Введите количество точек (8-12): "))
    if n < 5 or n > 12:
        print("Предупреждение: количество точек выходит за рекомендованный диапазон 8-12")
    x = []
    y = []
    for i in range(n):
        xi = float(input(f"x[{i+1}] = "))
        yi = float(input(f"y[{i+1}] = "))
        x.append(xi)
        y.append(yi)
    return np.array(x), np.array(y)


# Вывод результатов (с таблицей)
def print_results(model_name, params, S, delta, R2, x, y, y_pred, errors, r=None):
    print(" " * 60)
    print(f"Модель: {model_name}")
    
    if model_name == "Линейная":
        a, b = params
        print(f"Уравнение: y = {a:.6f} * x + {b:.6f}")
        if r is not None:
            print(f"Коэффициент корреляции Пирсона r = {r:.6f}")
    elif model_name == "Квадратичная":
        a0, a1, a2 = params
        print(f"Уравнение: y = {a0:.6f} + {a1:.6f}*x + {a2:.6f}*x^2")
    elif model_name == "Экспоненциальная":
        a, b = params
        print(f"Уравнение: y = {a:.6f} * exp({b:.6f} * x)")
    elif model_name == "Логарифмическая":
        a, b = params
        print(f"Уравнение: y = {a:.6f} * ln(x) + {b:.6f}")
    elif model_name == "Степенная":
        a, b = params
        print(f"Уравнение: y = {a:.6f} * x^{b:.6f}")
    
    print(f"\nМера отклонения S = {S:.6f}")
    print(f"Среднеквадратичное отклонение δ = {delta:.6f}")
    print(f"Коэффициент детерминации R² = {R2:.6f}")
    
    if R2 >= 0.95:
        print("  -> Высокая точность аппроксимации")
    elif R2 >= 0.75:
        print("  -> Удовлетворительная аппроксимация")
    elif R2 >= 0.5:
        print("  -> Слабая аппроксимация")
    else:
        print("  -> Точность недостаточна, модель требует изменения")
    
    # Вывод таблицы
    print_table(x, y, y_pred, errors, model_name)


# Сохранение результатов в файл (с таблицами)
def save_results_to_file(filename, results, x, y):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Результаты аппроксимации методом наименьших квадратов\n")
        f.write("="*60 + "\n")
        f.write(f"Исходные точки (n={len(x)}):\n")
        for xi, yi in zip(x, y):
            f.write(f"  x = {xi:.6f}, y = {yi:.6f}\n")
        f.write("\n")
        
        for (name, params, S, delta, R2, r, y_pred, errors) in results:
            f.write(f"\n{'='*60}\n")
            f.write(f"Модель: {name}\n")
            f.write(f"{'='*60}\n")
            
            if name == "Линейная":
                a, b = params
                f.write(f"Уравнение: y = {a:.6f} * x + {b:.6f}\n")
                if r is not None:
                    f.write(f"Коэффициент корреляции Пирсона r = {r:.6f}\n")
            elif name == "Квадратичная":
                a0, a1, a2 = params
                f.write(f"Уравнение: y = {a0:.6f} + {a1:.6f}*x + {a2:.6f}*x^2\n")
            elif name == "Экспоненциальная":
                a, b = params
                f.write(f"Уравнение: y = {a:.6f} * exp({b:.6f} * x)\n")
            elif name == "Логарифмическая":
                a, b = params
                f.write(f"Уравнение: y = {a:.6f} * ln(x) + {b:.6f}\n")
            elif name == "Степенная":
                a, b = params
                f.write(f"Уравнение: y = {a:.6f} * x^{b:.6f}\n")
            
            f.write(f"\nМера отклонения S = {S:.6f}\n")
            f.write(f"Среднеквадратичное отклонение δ = {delta:.6f}\n")
            f.write(f"Коэффициент детерминации R² = {R2:.6f}\n")
            
            # Сохраняем таблицу
            save_table_to_file(f, x, y, y_pred, errors, name)
        
        # Выбор лучшей модели
        best_names, best_delta = select_best_model([(r[0], r[3]) for r in results])
        
        if len(best_names) == 1:
            f.write(f"\nНаилучшая аппроксимирующая функция по критерию min δ: {best_names[0]} (δ = {best_delta:.6f})\n")
        else:
            f.write(f"\nНаилучшие аппроксимирующие функции (несколько с одинаковым СКО {best_delta:.6f}):\n")
            for name in best_names:
                f.write(f"  - {name}\n")


# Построение графиков
def plot_all(x, y, results, x_range_padding=0.1):
    """Построить графики исходных точек и всех аппроксимирующих функций"""
    plt.figure(figsize=(10, 6))
    # Исходные точки
    plt.scatter(x, y, color='black', label='Исходные точки', zorder=5)
    
    # Создаем гладкую сетку для построения кривых
    x_min, x_max = np.min(x), np.max(x)
    x_pad = (x_max - x_min) * x_range_padding
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 200)
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    color_idx = 0
    for (name, params, _, _, _, _, _, _) in results:
        if name == "Линейная":
            a, b = params
            y_grid = a * x_grid + b
        elif name == "Квадратичная":
            a0, a1, a2 = params
            y_grid = a0 + a1 * x_grid + a2 * x_grid**2
        elif name == "Экспоненциальная":
            a, b = params
            y_grid = a * np.exp(b * x_grid)
        elif name == "Логарифмическая":
            a, b = params
            mask = x_grid > 0
            y_grid = np.full_like(x_grid, np.nan)
            y_grid[mask] = a * np.log(x_grid[mask]) + b
        elif name == "Степенная":
            a, b = params
            mask = x_grid > 0
            y_grid = np.full_like(x_grid, np.nan)
            y_grid[mask] = a * (x_grid[mask] ** b)
        else:
            continue
        plt.plot(x_grid, y_grid, color=colors[color_idx % len(colors)], linestyle='-', linewidth=1.5, label=name)
        color_idx += 1
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация методом наименьших квадратов')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    print("Метод наименьших квадратов - аппроксимация табличных данных")
    print("Выберите способ ввода данных:")
    print("1 - из файла")
    print("2 - ввод с клавиатуры")
    choice = input("Ваш выбор (1/2): ").strip()

    if choice == '1':
        filename = input("Имя файла (например, data.txt): ").strip()
        try:
            x, y = read_data_from_file(filename)
        except Exception as e:
            print(f"Ошибка: {e}")
            return
    elif choice == '2':
        x, y = read_data_from_console()
    else:
        print("Неверный выбор")
        return

    if len(x) != len(y):
        print("Ошибка: длины массивов x и y не совпадают")
        return

    print("\nВыберите способ вывода результатов:")
    print("1 - в консоль")
    print("2 - в файл")
    output_choice = input("Ваш выбор (1/2): ").strip()

    if output_choice not in ['1', '2']:
        print("Неверный выбор, по умолчанию: консоль")
        output_choice = '1'

    if output_choice == '1':
        print(f"\nЗагружено {len(x)} точек:")
        for xi, yi in zip(x, y):
            print(f"  ({xi}, {yi})")

    results = []

    # Линейная
    try:
        a, b, S, delta, R2, r, y_pred, errors = linear_approximation(x, y)
        results.append(("Линейная", (a, b), S, delta, R2, r, y_pred, errors))
        if output_choice == '1':
            print_results("Линейная", (a, b), S, delta, R2, x, y, y_pred, errors, r)
    except Exception as e:
        print(f"Линейная аппроксимация не удалась: {e}")

    # Квадратичная
    try:
        a0, a1, a2, S, delta, R2, y_pred, errors = quadratic_approximation(x, y)
        results.append(("Квадратичная", (a0, a1, a2), S, delta, R2, None, y_pred, errors))
        if output_choice == '1':
            print_results("Квадратичная", (a0, a1, a2), S, delta, R2, x, y, y_pred, errors)
    except Exception as e:
        print(f"Квадратичная аппроксимация не удалась: {e}")

    # Экспоненциальная
    try:
        a, b, S, delta, R2, y_pred, errors = exponential_approximation(x, y)
        results.append(("Экспоненциальная", (a, b), S, delta, R2, None, y_pred, errors))
        if output_choice == '1':
            print_results("Экспоненциальная", (a, b), S, delta, R2, x, y, y_pred, errors)
    except Exception as e:
        print(f"Экспоненциальная аппроксимация не удалась: {e}")

    # Логарифмическая
    try:
        a, b, S, delta, R2, y_pred, errors = logarithmic_approximation(x, y)
        results.append(("Логарифмическая", (a, b), S, delta, R2, None, y_pred, errors))
        if output_choice == '1':
            print_results("Логарифмическая", (a, b), S, delta, R2, x, y, y_pred, errors)
    except Exception as e:
        print(f"Логарифмическая аппроксимация не удалась: {e}")

    # Степенная
    try:
        a, b, S, delta, R2, y_pred, errors = power_approximation(x, y)
        results.append(("Степенная", (a, b), S, delta, R2, None, y_pred, errors))
        if output_choice == '1':
            print_results("Степенная", (a, b), S, delta, R2, x, y, y_pred, errors)
    except Exception as e:
        print(f"Степенная аппроксимация не удалась: {e}")

    if not results:
        print("Ни одна модель не была успешно построена.")
        return

    best_names, best_delta = select_best_model([(r[0], r[3]) for r in results])

    if output_choice == '1':
        if len(best_names) == 1:
            print(f"Лучшая функция: {best_names[0]} (δ = {best_delta:.6f})")
        else:
            print(f"\n{'='*60}")
            print(f"Лучшие функции (несколько с одинаковым СКО {best_delta:.6f}):")
            for name in best_names:
                print(f"{name}")

    if output_choice == '2':
        out_filename = input("Введите имя файла (например results.txt): ").strip()
        save_results_to_file(out_filename, results, x, y)
        print(f"Результаты сохранены в файл {out_filename}")

    plot_all(x, y, results)


if __name__ == "__main__":
    main()