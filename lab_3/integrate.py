import numpy as np
import matplotlib.pyplot as plt

def create_function(func_str):
    # Allowed names
    allowed_names = {'sin': np.sin,
                     'cos': np.cos,
                     'tan': np.tan,
                     'exp': np.exp,
                     'sqrt': np.sqrt,
                     'log': np.log,
                     'log10': np.log10,
                     'pi': np.pi,
                     'e': np.e,
                     'arcsin': np.arcsin,
                     'arccos': np.arccos,
                     'arctan': np.arctan,
                     'sinh': np.sinh,
                     'cosh': np.cosh,
                     'tanh': np.tanh,
                     'abs': np.abs,
                     'np': np}

    def f(x, t=None):
        # Create a local dictionary with x and t
        local_dict = {'x': x, 't': t}
        # Update local_dict with allowed names
        local_dict.update(allowed_names)
        return eval(func_str, {"__builtins__": None}, local_dict)
    return f


# ф-ция возвращает порядок точности. для метода Рунге
def method_order(method):
    if method in ['left', 'right']:
        return 1
    elif method in ['midpoint', 'trapezoidal']:
        return 2
    elif method == 'simpson':
        return 4
    else:
        raise ValueError("Unknown method")


#Параметры:
#method: название метода.
#f: функция f(x,t)
#a, b: границы интегрирования.
#N: число разбиений.
#t: параметр функции.
def integrate(method, f, a, b, N, t):
    if method == 'left':
        return left_rectangles(f, a, b, N, t)
    elif method == 'right':
        return right_rectangles(f, a, b, N, t)
    elif method == 'midpoint':
        return midpoint_rectangles(f, a, b, N, t)
    elif method == 'trapezoidal':
        return trapezoidal(f, a, b, N, t)
    elif method == 'simpson':
        return simpson(f, a, b, N, t)
    else:
        raise ValueError("Unknown method")

def left_rectangles(f, a, b, N, t):
    h = (b - a) / N
    x = np.linspace(a, b - h, N)
    return h * np.sum(f(x, t))

def right_rectangles(f, a, b, N, t):
    h = (b - a) / N
    x = np.linspace(a + h, b, N)
    return h * np.sum(f(x, t))

def midpoint_rectangles(f, a, b, N, t):
    h = (b - a) / N
    x = np.linspace(a + h/2, b - h/2, N)
    return h * np.sum(f(x, t))

def trapezoidal(f, a, b, N, t):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x, t)
    return (h/2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

def simpson(f, a, b, N, t):
    if N % 2 == 1:
        N += 1  # Simpson's rule requires an even number of intervals
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x, t)
    return (h/3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

def adaptive_integration(method, f, a, b, eps, t):
    N = 10  # initial number of intervals
    p = method_order(method)
    I_N = integrate(method, f, a, b, N, t)
    N *= 2
    I_2N = integrate(method, f, a, b, N, t)
    error = abs(I_2N - I_N) / (2**p - 1)
    while error > eps:
        I_N = I_2N
        N *= 2
        I_2N = integrate(method, f, a, b, N, t)
        error = abs(I_2N - I_N) / (2**p - 1)
        if N > 1e7:
            print("Достигнуто максимальное число разбиений. Интеграл может не сходиться.")
            break
    return I_2N

def improper_integral(method, f, a, eps):
    B = a + 1  # Initial upper limit
    I_B = adaptive_integration(method, f, a, B, eps, t=None)
    B *= 2
    I_2B = adaptive_integration(method, f, a, B, eps, t=None)
    error = abs(I_2B - I_B)
    while error > eps:
        I_B = I_2B
        B *= 2
        I_2B = adaptive_integration(method, f, a, B, eps, t=None)
        error = abs(I_2B - I_B)
        if np.isinf(I_2B) or np.isnan(I_2B) or B > 1e10:
            print("Интеграл расходится.")
            return None
    print(f"Интеграл сходится к значению: {I_2B}")
    return I_2B

def main():
    # Integration interval [a, b]
    a = float(input("Введите нижний предел интегрирования a: "))
    b = float(input("Введите верхний предел интегрирования b: "))

    # t range [alpha, beta]
    alpha = float(input("Введите начало интервала параметра t (α): "))
    beta = float(input("Введите конец интервала параметра t (β): "))

    # Precision ε
    eps = float(input("Введите точность вычисления ε: "))

    # Integration method
    print("Выберите метод численного интегрирования:")
    print("1 - Левые прямоугольники")
    print("2 - Правые прямоугольники")
    print("3 - Средние прямоугольники")
    print("4 - Трапеции")
    print("5 - Симпсон")

    method_choice = input("Введите номер метода (1-5): ")

    method_dict = {
        '1': 'left',
        '2': 'right',
        '3': 'midpoint',
        '4': 'trapezoidal',
        '5': 'simpson'
    }

    method = method_dict.get(method_choice)
    if method is None:
        print("Неверный выбор метода.")
        return

    # Function f(x, t)
    predefined_functions = [
        ('sin(x * t)', 'np.sin(x * t)'),
        ('exp(-x * t)', 'np.exp(-x * t)'),
        ('x ** t', 'x ** t'),
        ('cos(x) * sin(t)', 'np.cos(x) * np.sin(t)'),
        ('1 / (x + t)', '1 / (x + t)'),
    ]

    print("Выберите функцию для интегрирования:")
    for i, (desc, func_str) in enumerate(predefined_functions, start=1):
        print(f"{i} - {desc}")
    print(f"{len(predefined_functions)+1} - Ввести свою функцию")

    func_choice = input(f"Введите номер функции (1-{len(predefined_functions)+1}): ")

    if func_choice.isdigit():
        func_choice = int(func_choice)
        if 1 <= func_choice <= len(predefined_functions):
            func_str = predefined_functions[func_choice - 1][1]
        elif func_choice == len(predefined_functions)+1:
            func_str = input("Введите функцию f(x, t), используя синтаксис numpy: ")
        else:
            print("Неверный выбор функции.")
            return
    else:
        print("Неверный ввод.")
        return

    # Create the function f(x, t)
    f = create_function(func_str)

    # Compute I(t) over t in [alpha, beta]
    num_t_points = 100
    t_values = np.linspace(alpha, beta, num_t_points)
    I_values = np.zeros_like(t_values)

    print("Вычисление интегралов...")
    for i, t in enumerate(t_values):
        I_values[i] = adaptive_integration(method, f, a, b, eps, t)
        print(f"t = {t:.5f}, I(t) = {I_values[i]:.5f}")

    # Plot I(t) vs t
    plt.figure()
    plt.plot(t_values, I_values)
    plt.xlabel('t')
    plt.ylabel('I(t)')
    plt.title('График функции I(t)')
    plt.grid(True)
    plt.show()

    # Optional improper integral
    improper_choice = input("Хотите вычислить несобственный интеграл от a до бесконечности? (y/n): ")
    if improper_choice.lower() == 'y':
        # Function f(x)
        func_str_improper = input("Введите функцию f(x), используя синтаксис numpy: ")
        f_improper = create_function(func_str_improper)
        result = improper_integral(method, f_improper, a, eps)
        if result is not None:
            print(f"Значение интеграла: {result}")
        else:
            print("Интеграл расходится.")

if __name__ == "__main__":
    main()
