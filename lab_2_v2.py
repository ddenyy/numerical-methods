import numpy as np
import matplotlib.pyplot as plt


def gradient_base(f, x, h=1e-5):
    grad = np.zeros_like(x)  # Вектор для хранения градиента
    
    # Для каждой переменной x[i], вычисляем частную производную
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_forward[i] += h  # Вносим небольшое изменение в одну переменную
        grad[i] = (f(x_forward) - f(x)) / h  # Приближенная производная по переменной x[i]
    
    return grad


# Определение целевой функции (например, квадратичная)
def objective_function(x):
    return np.sin(x[0]) * np.cos(x[1]/2)
    #return x[1] ** 2 + 2 * x[0] ** 2

# Метод золотого сечения для одномерной оптимизации
def golden_section_search(func, a, b, tol=1e-5, max_iter=20):
    gr = (np.sqrt(5) + 1) / 2  # Золотое сечение
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    iter_count = 0

    while abs(b - a) > tol and iter_count < max_iter:
        if func(c) < func(d):
            b = d
        else:
            a = c

        # Обновляем точки c и d
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        iter_count += 1

    return (b + a) / 2

# Метод покоординатного спуска с использованием метода золотого сечения
def coordinate_descent(starting_point, tol=1e-5, iterations=100):
    x = np.array(starting_point, dtype=float)
    history = [x.copy()]

    for i in range(iterations):
        for j in range(len(x)):  # Оптимизация по каждой координате
            # Определяем функцию одной переменной phi(s) для оптимизации по координате x[j]
            def phi(s):
                x_temp = x.copy()
                x_temp[j] = s
                return objective_function(x_temp)
            
            # Устанавливаем начальный отрезок для поиска минимума
            a = x[j] - 5  # Можно выбрать другие значения в зависимости от задачи
            b = x[j] + 5

            # Используем метод золотого сечения для нахождения минимума по координате x[j]
            x_opt = golden_section_search(phi, a, b, tol=tol)
            x[j] = x_opt  # Обновляем координату x[j]

            history.append(x.copy())

    return x, history

# Наискорейший градиентный спуск
def steepest_descent(starting_point, tol=1e-5, iterations=1000):
    x = np.array(starting_point, dtype=float)
    history = [x.copy()]

    def gradient(x):
        # """Градиент целевой функции."""
        # return np.array([np.cos(x[0]) * np.cos(x[1]/2), ((-1) * np.sin(x[0]) * np.sin(x[1]/2)) / 2 ])
        return gradient_base(objective_function, x)

    for i in range(iterations):
        grad = gradient(x)

        # Функция для поиска оптимального шага по направлению антиградиента
        def phi(alpha):
            return objective_function(x - alpha * grad)

        # Поиск оптимального шага с помощью метода золотого сечения
        alpha_opt = golden_section_search(phi, 0, 1, tol=tol)

        # Обновляем координаты с использованием найденного шага
        x_new = x - alpha_opt * grad
        history.append(x_new.copy())

        # Проверка критерия остановки
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, history

# Визуализация шагов спуска
def visualize_descent(history):
    x_values = [point[0] for point in history]
    y_values = [point[1] for point in history]
    
    X, Y = np.meshgrid(np.linspace(-10, 10, 600), np.linspace(-10, 10, 600))
    # Z = np.sin(X) + np.cos(Y/2)  # Целевая функция для визуализации
    Z = objective_function([X, Y])

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, cmap='jet')
    plt.plot(x_values, y_values, 'bo-', markersize=5, label='Путь спуска')
    plt.scatter(x_values[0], y_values[0], color='red', label='Начало')
    plt.scatter(x_values[-1], y_values[-1], color='green', label='Конец')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Путь к минимуму')
    plt.legend()
    plt.show()

# Запрос ввода данных у пользователя
try:
    tol = float(input("Введите точность (ε): "))
    x0 = float(input("Введите начальную координату x0: "))
    y0 = float(input("Введите начальную координату y0: "))
    print("Выберите метод вычисления минимума:")
    print("1 - Покоординатный спуск с методом золотого сечения")
    print("2 - Наискорейший градиентный спуск")
    method = int(input("Введите номер метода (1 или 2): "))

    starting_point = [x0, y0]

    # Выбор метода
    if method == 1:
        final_point, history = coordinate_descent(starting_point, tol=tol)
        print(f"Минимум найден с покоординатным спуском: {final_point}")
    elif method == 2:
        final_point, history = steepest_descent(starting_point, tol=tol)
        print(f"Минимум найден с наискорейшим градиентным спуском: {final_point}")
    else:
        print("Неверный выбор метода.")
        exit()

    # Визуализация
    visualize_descent(history)

except ValueError:
    print("Ошибка ввода. Убедитесь, что вы ввели числовые значения.")
