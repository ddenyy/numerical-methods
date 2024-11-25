import numpy as np
import matplotlib.pyplot as plt

def simple_iteration(A, b, tau, epsilon=1e-5, max_iterations=100000):
    """
    Решение системы Ax = b методом простой итерации.

    Параметры:
    A : np.ndarray
        Коэффициентная матрица.
    b : np.ndarray
        Вектор правой части.
    tau : float
        Параметр масштаба для построения итерационного процесса.
    epsilon : float
        Точность решения.
    max_iterations : int
        Максимальное число итераций.

    Возвращает:
    x : np.ndarray
        Найденное решение.
    """
    n = len(b)
    x = np.zeros(n)  # Начальное приближение
    B = np.eye(n) - tau * A
    c = tau * b


    for iteration in range(max_iterations):
        x_new = B @ x + c
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            print(f"Сходится за {iteration + 1} итераций")
            return x_new
        x = x_new

    raise ValueError("Метод не сходится за максимальное число итераций.")

def read_data(filename):
    """
    Читает данные из файла.

    Формат файла:
    x1 y1
    x2 y2
    ...

    Возвращает:
    x_data : np.ndarray
        Массив x_i.
    y_data : np.ndarray
        Массив y_i.
    """
    data = np.loadtxt(filename)
    x_data = data[:, 0]
    y_data = data[:, 1]
    return x_data, y_data

def construct_system(x_data, y_data, m):
    """
    Формирует систему нормальных уравнений для полиномиальной регрессии степени m.

    Параметры:
    x_data : np.ndarray
        Массив x_i.
    y_data : np.ndarray
        Массив y_i.
    m : int
        Степень многочлена.

    Возвращает:
    A : np.ndarray
        Коэффициентная матрица системы.
    b : np.ndarray
        Вектор правой части.
    """
    n = len(x_data)
    # Инициализация матриц
    A = np.zeros((m + 1, m + 1))
    b = np.zeros(m + 1)

    # Заполнение матрицы A и вектора b
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x_data ** (i + j))
        b[i] = np.sum(y_data * (x_data ** i))

    return A, b

def polynomial(x, coeffs):
    """
    Вычисляет значение многочлена в точке x.

    Параметры:
    x : float или np.ndarray
        Точка или массив точек.
    coeffs : np.ndarray
        Коэффициенты многочлена.

    Возвращает:
    y : float или np.ndarray
        Значение многочлена в точке x.
    """
    y = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coeffs):
        y += a * x ** i
    return y

def main():
    # Чтение данных из файла
    x_data, y_data = read_data("/Users/denisosipov/All_programs/numbers_method_kosterin/lab_4/input_noise.txt")

    # Ввод степени многочлена
    m = int(input("Введите степень многочлена m: "))

    # Формирование системы нормальных уравнений
    A, b = construct_system(x_data, y_data, m)

    # Подбор параметра tau для обеспечения сходимости
    # Обычно выбирают tau = 1 / ||A||
    tau = 1 / np.linalg.norm(A, ord=np.inf)
    print(f"Выбранный tau: {tau}")

    # Решение системы методом простой итерации
    try:
        coeffs = simple_iteration(A, b, tau)
        print("Коэффициенты многочлена:", coeffs)
    except ValueError as e:
        print(str(e))
        return

    # Построение графика
    x_plot = np.linspace(np.min(x_data), np.max(x_data), 500)
    y_plot = polynomial(x_plot, coeffs)

    plt.scatter(x_data, y_data, color='blue', label='Данные')
    plt.plot(x_plot, y_plot, color='red', label='Многочлен степени {}'.format(m))
    plt.legend()
    plt.title('Полиномиальная регрессия степени {}'.format(m))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
