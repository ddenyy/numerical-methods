Конечно! Вот примеры того, как вы можете использовать программу для вычисления несобственных интегралов. Я предоставлю пошаговое руководство и объяснения.

---

### **Пример 1: Вычисление сходящегося несобственного интеграла**

**Интеграл:**

\[
\int\_{1}^{+\infty} \frac{1}{x^2} \, dx
\]

Этот интеграл известен и сходится к значению 1.

**Шаги для вычисления:**

1. **Запустите программу** и следуйте инструкциям.

2. **Введите нижний предел интегрирования `a`:**

   ```
   Введите нижний предел интегрирования a: 1
   ```

3. **Введите верхний предел интегрирования `b`:**

   ```
   Введите верхний предел интегрирования b: 2
   ```

   Поскольку мы будем интегрировать до бесконечности, значение `b` здесь не имеет значения, но программа требует ввода этого параметра.

4. **Введите интервал параметра `t`:**

   Для несобственного интеграла параметр `t` не используется, но программа требует ввода:

   ```
   Введите начало интервала параметра t (α): 0
   Введите конец интервала параметра t (β): 0
   ```

5. **Введите точность `ε`:**

   ```
   Введите точность вычисления ε: 0.0001
   ```

6. **Выберите метод численного интегрирования:**

   ```
   Выберите метод численного интегрирования:
   1 - Левые прямоугольники
   2 - Правые прямоугольники
   3 - Средние прямоугольники
   4 - Трапеции
   5 - Симпсон
   Введите номер метода (1-5): 5
   ```

   Рекомендуется использовать метод Симпсона для лучшей точности.

7. **Выберите функцию для интегрирования:**

   ```
   Выберите функцию для интегрирования:
   1 - sin(x * t)
   2 - exp(-x * t)
   3 - x ** t
   4 - cos(x) * sin(t)
   5 - 1 / (x + t)
   6 - Ввести свою функцию
   Введите номер функции (1-6): 6
   ```

   Мы выберем опцию 6, чтобы ввести свою функцию.

8. **Введите функцию \( f(x) \):**

   ```
   Введите функцию f(x, t), используя синтаксис numpy: 1 / x ** 2
   ```

9. **Пропуск вычисления \( I(t) \):**

   Поскольку \( t \) не используется, программа быстро завершит этот шаг.

10. **Выберите вычисление несобственного интеграла:**

    ```
    Хотите вычислить несобственный интеграл от a до бесконечности? (y/n): y
    ```

11. **Введите функцию для несобственного интеграла:**

    ```
    Введите функцию f(x), используя синтаксис numpy: 1 / x ** 2
    ```

12. **Результат:**

    Программа вычислит интеграл и выведет:

    ```
    Интеграл сходится к значению: 1.0000
    Значение интеграла: 1.0000
    ```

    Это соответствует аналитическому решению:

    \[
    \int\_{1}^{+\infty} \frac{1}{x^2} \, dx = 1
    \]

---

### **Пример 2: Вычисление расходящегося несобственного интеграла**

**Интеграл:**

\[
\int\_{1}^{+\infty} \frac{1}{x} \, dx
\]

Этот интеграл расходится (стремится к бесконечности).

**Шаги для вычисления:**

1. **Введите нижний предел интегрирования `a`:**

   ```
   Введите нижний предел интегрирования a: 1
   ```

2. **Введите верхний предел интегрирования `b`:**

   ```
   Введите верхний предел интегрирования b: 2
   ```

3. **Введите интервал параметра `t`:**

   ```
   Введите начало интервала параметра t (α): 0
   Введите конец интервала параметра t (β): 0
   ```

4. **Введите точность `ε`:**

   ```
   Введите точность вычисления ε: 0.0001
   ```

5. **Выберите метод численного интегрирования:**

   ```
   Введите номер метода (1-5): 5
   ```

6. **Введите функцию \( f(x) \):**

   ```
   Введите номер функции (1-6): 6
   Введите функцию f(x, t), используя синтаксис numpy: 1 / x
   ```

7. **Выберите вычисление несобственного интеграла:**

   ```
   Хотите вычислить несобственный интеграл от a до бесконечности? (y/n): y
   ```

8. **Введите функцию для несобственного интеграла:**

   ```
   Введите функцию f(x), используя синтаксис numpy: 1 / x
   ```

9. **Результат:**

   ```
   Интеграл расходится.
   ```

   Программа определила, что интеграл не сходится, что соответствует теоретическому результату.

---

### **Пример 3: Вычисление несобственного интеграла с параметром**

**Интеграл:**

\[
\int\_{0}^{+\infty} e^{-k x} \, dx
\]

Будем считать \( k > 0 \) параметром и вычислим интеграл для нескольких значений \( k \).

**Шаги для вычисления:**

1. **Введите нижний предел интегрирования `a`:**

   ```
   Введите нижний предел интегрирования a: 0
   ```

2. **Введите верхний предел интегрирования `b`:**

   ```
   Введите верхний предел интегрирования b: 1
   ```

3. **Введите интервал параметра `t`:**

   Пусть \( t = k \) меняется от 1 до 5.

   ```
   Введите начало интервала параметра t (α): 1
   Введите конец интервала параметра t (β): 5
   ```

4. **Введите точность `ε`:**

   ```
   Введите точность вычисления ε: 0.0001
   ```

5. **Выберите метод численного интегрирования:**

   ```
   Введите номер метода (1-5): 5
   ```

6. **Выберите функцию для интегрирования:**

   Выберем функцию \( f(x, t) = \exp(-t \cdot x) \):

   ```
   Введите номер функции (1-6): 2
   ```

   Это соответствует функции `exp(-x * t)`.

7. **Вычисление \( I(t) \):**

   Программа начнет вычислять \( I(t) \) для \( t \) от 1 до 5 и построит график.

8. **Выберите вычисление несобственного интеграла:**

   ```
   Хотите вычислить несобственный интеграл от a до бесконечности? (y/n): y
   ```

9. **Введите функцию для несобственного интеграла:**

   Поскольку параметр \( t \) не поддерживается в этой части программы, выберем конкретное значение \( t \). Допустим, \( t = 2 \):

   ```
   Введите функцию f(x), используя синтаксис numpy: exp(-2 * x)
   ```

10. **Результат:**

    ```
    Интеграл сходится к значению: 0.5000
    Значение интеграла: 0.5000
    ```

    Аналитически:

    \[
    \int\_{0}^{+\infty} e^{-2 x} \, dx = \frac{1}{2}
    \]

    Таким образом, программа правильно вычислила интеграл.

---

### **Примечания по использованию программы:**

- **Ввод функции:**

  - При вводе функций используйте синтаксис библиотеки NumPy.
  - Например, для \( \frac{1}{x^2} \) вводите `1 / x ** 2`.
  - Для экспоненты \( e^{-x} \) вводите `exp(-x)`.

- **Избегайте ошибок:**

  - Убедитесь, что функция определена на всём интервале интегрирования.
  - Если функция имеет особенности (полюсы) внутри интервала, результаты могут быть неверными.

- **Проверка сходимости:**

  - Программа использует критерий сходимости:

    \[
    |I\_{2B} - I_B| < \varepsilon
    \]

    где \( I*B \) — значение интеграла на интервале \([a, B]\), а \( I*{2B} \) — на \([a, 2B]\).

- **Ограничения:**

  - Если интеграл не сходится или вычисления требуют очень большого количества разбиений, программа сообщит об этом.
  - При очень малых значениях \( \varepsilon \) вычисления могут занимать длительное время.

---

### **Заключение**

Программа позволяет эффективно вычислять несобственные интегралы и проверять их на сходимость. Она может быть полезна для учебных целей или при решении практических задач, где требуется численное интегрирование на бесконечных интервалах.

Если у вас возникнут вопросы или нужны дополнительные пояснения по использованию программы, не стесняйтесь спрашивать!
