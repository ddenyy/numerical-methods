

import BisectionMethod
from NewtonMethod import NewtonSolver

def display_menu():
    """Отображает меню выбора метода решения."""
    print("==== Метод решения ====")
    print("1) Метод Ньютона")
    print("2) Метод деления пополам")
    print("0) Выход")
    print("========================")

def newton_method_start():
    """Запрашивает ввод переменных для метода Ньютона."""
    a = float(input("Введите значение a: "))
    b = float(input("Введите значение b: "))
    alpha = float(input("Введите значение alpha: "))
    betta = float(input("Введите значение betta: "))
    print(f"Вы ввели: a = {a}, b = {b}, alpha = {alpha}, betta = {betta}")

    # Создаем экземпляр класса и запускаем решение
    solver = NewtonSolver(a, b, alpha, betta)
    solver.find_solutions()
    solver.plot_solutions()


def bisection_method_start():
  """Запрашивает ввод переменных для метода деления отрезка пополам."""
  a = float(input("Введите значение a: "))
  b = float(input("Введите значение b: "))
  alpha = float(input("Введите значение alpha: "))
  betta = float(input("Введите значение betta: "))
  print(f"Вы ввели: a = {a}, b = {b}, alpha = {alpha}, betta = {betta}")

  bisection_solver = BisectionMethod(a, b, alpha, betta)
  bisection_solver.find_solutions()
  bisection_solver.plot_solutions()


def main():
    while True:
        display_menu()
        choice = input("Выберите метод (1/2/0): ")

        if choice == '1':
            newton_method_start()
        elif choice == '2':
            bisection_method_start()
        elif choice == '0':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Пожалуйста, попробуйте снова.")

if __name__ == "__main__":
    main()
