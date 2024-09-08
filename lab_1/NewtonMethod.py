import numpy as np
import matplotlib.pyplot as plt

class NewtonSolver:
    def __init__(self, a, b, alpha, betta, num_m=1000, num_x0=10):
        """Инициализация параметров."""
        self.a = a
        self.b = b
        self.alpha = alpha
        self.betta = betta
        self.m_values = np.linspace(alpha, betta, num_m)  # Различные значения параметра m
        self.x0_values = np.linspace(a, b, num_x0)  # Начальные значения x0
        self.solutions = []

    def f(self, x, m):
        """Определяем функцию f(x, m)."""
        return x**2 - m

    def f_prime(self, x, m):
        delta = 1e-9
        """Определяем производную f'(x, m)."""
        return (self.f(x + delta, m) - self.f(x, m)) / delta

    def newton_method(self, x0, m, tol=1e-9, max_iter=100):
        """Решение уравнения f(x, m) = 0 методом Ньютона."""
        x = x0
        for _ in range(max_iter):
            fx = self.f(x, m)
            fpx = self.f_prime(x, m)

            if abs(fpx) < 1e-9:  # Проверка на нулевую производную
                print("Производная равна нулю. Метод Ньютона не может быть применён.")
                return None

            x_new = x - fx / fpx

            if abs(x_new - x) < tol:  # Проверка на сходимость
                return x_new

            x = x_new

        print("Достигнуто максимальное количество итераций.")
        if self.a < x < self.b:
            return x
        else:
            return None

    def find_solutions(self):
        for x0 in self.x0_values:
            for m in self.m_values:
                solution = self.newton_method(x0, m)
                if solution is not None:
                    self.solutions.append((m, solution))

    def plot_solutions(self):
        """Решение уравнения для различных значений m и построение графика."""
        # Подготовка данных для графика
        m_solutions = [sol[0] for sol in self.solutions]
        x_solutions = [sol[1] for sol in self.solutions]
        plt.figure(figsize=(10, 6))
        plt.scatter(m_solutions, x_solutions, color="blue", s=10)
        plt.title('Решения уравнения f(x, m) = 0 для различных значений m')
        plt.xlabel('Значение параметра m')
        plt.ylabel('Корень уравнения x')
        plt.ylim(self.a, self.b)
        plt.xlim(self.alpha, self.betta)
        plt.axhline(0, color='gray', lw=0.5, ls='--')
        plt.axvline(0, color='gray', lw=0.5, ls='--')
        plt.grid()
        plt.show()


# Параметры
a = -2
b = 2
alpha = -3
betta = 3

# Создаем экземпляр класса и запускаем решение
solver = NewtonSolver(a, b, alpha, betta)
solver.find_solutions()
solver.plot_solutions()