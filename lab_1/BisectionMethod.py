from math import sin
import numpy as np
import matplotlib.pyplot as plt

class BisectionMethod:
    def __init__(self, a, b, alpha, betta, epsilon=1e-9, max_iter=100, num_subintervals=10, num_m=1000):
        self.a = a
        self.b = b
        self.m_values = np.linspace(alpha, betta, num_m)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.num_subintervals = num_subintervals
        self.solutions = []

    def f(self, x, m):
        """Определяем функцию f(x, m)."""
        return sin(x+m)  # Пример функции, измените по мере необходимости.

    def bisection(self, a, b, m):
        """Метод деления отрезка пополам для нахождения корня уравнения."""
        if self.f(a, m) * self.f(b, m) >= 0:
            return None  # Нет решения на данном отрезке

        for _ in range(self.max_iter):
            c = (a + b) / 2  # Находим середину отрезка
            if abs(self.f(c, m)) < self.epsilon:  # Если значение функции в середине достаточно мало
                return c  # Считаем c решением уравнения

            if self.f(a, m) * self.f(c, m) < 0:  # Если знак f(a) и f(c) разные
                b = c  # Решение находится в отрезке [a, c]
            else:
                a = c  # Решение находится в отрезке [c, b]

        return (a + b) / 2  # Возвращаем приближенное значение корня

    def find_solutions(self):
        """Ищем решения уравнения f(x, m) = 0 для различных значений m."""
        for m in self.m_values:
            # Разбиваем отрезок [a, b] на подотрезки
            sub_intervals = np.linspace(self.a, self.b, self.num_subintervals + 1)
            for i in range(len(sub_intervals) - 1):
                a_sub = sub_intervals[i]
                b_sub = sub_intervals[i + 1]

                solution = self.bisection(a_sub, b_sub, m)
                if solution is not None:
                    self.solutions.append((m, solution))

    def plot_solutions(self):
        """Создает график решений."""
        m_solutions = [sol[0] for sol in self.solutions]
        x_solutions = [sol[1] for sol in self.solutions]

        plt.figure(figsize=(10, 6))
        plt.scatter(m_solutions, x_solutions, color="blue", s=10)
        plt.title('Решения уравнения f(x, m) = 0 для различных значений m')
        plt.xlabel('Значение параметра m')
        plt.ylabel('Корень уравнения x')
        plt.axhline(0, color='gray', lw=0.5, ls='--')
        plt.axvline(0, color='gray', lw=0.5, ls='--')
        plt.grid()
        plt.show()

# Пример использования
if __name__ == "__main__":
  a = -2  # Левая граница
  b = 2   # Правая граница

  bisection_solver = BisectionMethod(a, b, -3, 3)
  bisection_solver.find_solutions()
  bisection_solver.plot_solutions()