# Метод главных компонент (Principal Component Analysis)
### Зачем:
1. Решает проблему мультиколлинеарности
1. Получает минимальное количество признаков из исходных
1. Определяет эффективную размерность исходных данных

### Задача:
* **x<sub>1 .. l</sub>** -- объекты;
* **l** -- количество объектов;
* **f<sub>1 .. n</sub>** -- числовые признаки объектов;
* **n** -- количество признаков каждого объекта;

Матрица **F**<sub>l x n</sub> -- матрица признаков объектов. F[i][j] = j-ый признак i-го элемента: **f<sub>j</sub>(x<sub>i</sub>)**.

Матрица **G**<sub>l x m</sub> -- новая матрица описания признаков объектов, при этом **m < n**. Тогда G[i][j] = g<sub>j</sub>(x<sub>i</sub>). Новые объекты обозначим z<sub>1 .. l</sub>.

Исходные признаки можно восстановить из **z** линейным преобразованием. **x<sup>^</sup> = z * U<sup>T</sup>**. x<sup>^</sup> должно минимально отличаться от x при выбранном m.

Необходимо найти такие G и U.

![](pca_formula.png)

||A||<sup>2</sup> = trAA<sup>T</sup> = trA<sup>T</sup>A, где tr -- след матрицы (сумма элементов главной диагонали). Ранг (rk) G и U равен m.

### Теорема
Если m <= rk F, то минимум ∆<sup>2</sup>(G, U) достигается, когда столбцы матрицы U есть собственные векторы F<sup>T</sup>F, соответствующие m максимальным собственным значениям. При этом G = FU, матрицы U и G ортогонмальны.

## Связь с сингулярным разложением
Если **m = n**, то **∆<sup>2</sup>(G, U) = 0**.
В этом случае представление **F = GU<sup>T</sup>** является точным и совпадает с сингулярным разложением: **F = GU<sup>T</sup> = VDU<sup>T</sup>**, если положить **G = VD** и **Λ = D2**. При этом матрица V ортогональна: **V<sup>T</sup>V = I<sub>m</sub>**.

Если **m < n**, то представление **F≈GU<sup>T</sup>** является приближённым.
Сингулярное разложение матрицы GU<sup>T</sup> получается из сингулярного разложения матрицы F путём обнуления **n − m** минимальных собственных значений.

## Преобразование Карунена–Лоэва.
Диагональность матрицы **G<sup>T</sup>G = Λ** означает, что новые признаки **g<sub> 1 .. m</sub>** не коррелируют на обучающих объектах.
Ортогональное преобразование U называют декоррелирующим или преобразованием Карунена–Лоэва.
Если **m = n**, то прямое и обратное преобразование вычисляются с помощью одной и той же матрицы U: **F = GU<sup>T</sup>** и **G = FU**.

## Эффективная размерность.
Главные компоненты содержат основную информацию о матрице F.
Число главных компонент m называют также эффективной размерностью задачи.
На практике её определяют следующим образом.
Все собственные значения матрицы **F<sup>T</sup>F** упорядочиваются по убыванию:
**λ<sub>1</sub> >= ... >= λ<sup>n</sup> >= 0**.
Задаётся пороговое значение ε из **[0, 1]**, достаточно близкое к нулю, и определяется наименьшее целое m, при котором относительная погрешность приближения матрицы F не превышает ε:

![](pca_err.png)