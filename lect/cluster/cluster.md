# Кластеризация
## Задача кластеризации
Имеется:
- обучающая выборка **X<sup>l</sup> = {x<sub>1</sub>, .., x<sub>l</sub>}** из X
- функция расстояния между объектами **ρ(x, x')**

Требуется:
- разбить выборку на непересекающиеся подмножества, называемые кластерами,
так, чтобы каждый кластер состоял из объектов, близких по метрике ρ, а объекты разных кластеров существенно отличались
- при этом каждому объекту **x<sub>i</sub>** из **X<sup>l</sup>** приписывается номер кластера **y<sub>i</sub>**

## Определение
Алгоритм кластеризации -- это функция **a: X -> Y** , которая
любому объекту **x из X** ставит в соответствие метку кластера **y ∈ Y**.
Множество меток Y в некоторых случаях известно заранее, однако чаще ставится задача определить оптимальное число кластеров, с точки зрения того или иного критерия качества кластеризации.

    Не существует однозначно наилучшего критерия качества кластеризации

## Цели кластеризации
- Упростить дальнейшую обработку данных, разбить множество **X<sup>ℓ</sup>**
на группы схожих объектов чтобы работать с каждой группой в отдельности
*(задачи классификации, регрессии, прогнозирования)*
- Сократить объём хранимых данных, оставив по одному представителюот каждого кластера
*(задачи сжатия данных)*
- Выделить нетипичные объекты, которые не подходят ни к одному из кластеров
*(задачи одноклассовой классификации)*
- Построить иерархию множества объектов
*(задачи таксономии)*

## Типы кластерных структур

| Пример | Описание |
|----------------|-|
|![](type_01.png)|**Сгущения**: внутрикластерные расстояния, как правило, меньше межкластерных|
|![](type_02.png)|**Ленты**: для любого объекта найдётся близкий к нему объект того же кластера, в то же время существуют объекты одного кластера, которые не являются близкими|
|![](type_03.png)|**Кластеры с центром**: в каждом кластере найдётся объект, такой, что почти все объекты кластера лежат внутри шара с центром в этом объекте|
|![](type_04.png)|Кластеры могут соединяться перемычками, что затрудняет работу многих алгоритмов кластеризации|
|![](type_05.png)|Кластеры могут накладываться на разреженный фон из редких нетипичных объектов|
|![](type_06.png)|Кластеры могут перекрываться|
|![](type_07.png)|Кластеры могут образовываться не по принципу сходства, а по каким-либо иным, заранее неизвестным, свойствам объектов. Стандартные методы кластеризации здесь бессильны|
|![](type_08.png)|Кластеры могут вообще отсутствовать. В этом случае надо применять не кластеризацию, а иные методы анализа данных|

## Эврестические графовые алгоритмы
Обширный класс алгоритмов кластеризации основан на представлении выборки в виде графа.
Вершинам графа соответствуют объекты выборки, а рёбрам -- попарные расстояния
между объектами **ρ<sub>ij</sub> = ρ(x<sub>i</sub>, x<sub>j</sub>)**.

### Алгоритм выделения связных компонент
Задаётся параметр **R** и в графе удаляются все рёбра (i, j),
для которых **ρ<sub>ij</sub> > R**.
Соединёнными остаются только наиболее близкие пары объектов.
Идея алгоритма заключается в том, чтобы подобрать такое значение
**R ∈ [min ρ<sub>ij</sub> , max ρ<sub>ij</sub>]**,
при котором граф развалится на несколько связных компонент.
Найденные связные компоненты -- и есть кластеры.

*Связной компонентой графа* называется подмножество его вершин,
в которомлюбые две вершины можно соединить путём, целиком лежащим в этом подмножестве.
Для поиска связных компонент можно использовать стандартные
алгоритмы поиска в ширину (алгоритм Дейкстры) или поиска в глубину.

### Алгоритм кратчайшего незамкнутого пути
Алгоритм строит граф из l−1 рёбер так, чтобы они соединяли все l точек
и обладали минимальной суммарной длиной.
Такой граф называется кратчайшим незамкнутым путём (КНП),
минимальным покрывающим деревом или каркасом.
Доказано, что этот граф строится с помощью несложной процедуры,
соответствующей шагам 1–4 [Алгоритма выделения связных компонент](#алгоритм-выделения-связных-компонент).
На шаге 5 удаляются K − 1 самых длинных рёбер, и связный граф распадается на K кластеров.

В отличие от предыдущего алгоритма, число кластеров K задаётся как входной параметр.
Его можно также определять графически, если упорядочить все расстояния,
образующие каркас, в порядке убывания и отложить их на графике.
Резкий скачок вниз где-то на начальном (левом) участке графика покажет
количество наиболее чётко выделяемых кластеров.

Этот алгоритм, как и предыдущий, очень прост и также имеет ограниченную применимость.
Наличие разреженного фона или перемычек приводит к неадекватной кластеризации.
Другим недостатком КНП является высокая трудоёмкость --
для построения кратчайшего незамкнутого пути требуется **O(l<sup>3</sup>)** операций.

## Функционалы качества кластеризации
Существует много разновидностей функционалов качества кластеризации,
но нет "самого правильного" функционала.
Каждый метод кластеризации можно рассматривать как точный или приближённый
алгоритм поиска оптимума некоторого функционала.

Среднее внутрикластерное расстояние должно быть как можно меньше:

![Функционал 1](f_01.png)

Среднее межкластерное расстояние должно быть как можно больше:

![Функционал 2](f_02.png)

Если алгоритм кластеризации вычисляет центры кластеров **μ<sub>y</sub>, y ∈ Y**,
то можно определить функционалы, вычислительно более эффективные.

Сумма средних внутрикластерных расстояний должна быть как можно меньше:

![Функционал 3](f_03.png)

где **K<sub>y</sub> = {x<sub>i</sub> ∈ X<sup>l</sup> | y<sub>i</sub> = y}** -- кластер с номером y.

Сумма межкластерных расстояний должна быть как можно больше:

![Функционал 4](f_04.png)

где **μ** -- центр масс всей выборки.

На практике вычисляют отношение пары функционалов,
чтобы учесть как межкластерные, так и внутрикластерные расстояния:

**F<sub>0</sub>/F<sub>1</sub> -> min, либо Ф<sub>0</sub>/Ф<sub>1</sub> -> min**

## Статические алгоритмы
Статистические алгоритмы основаны на предположении, что
кластеры неплохо описываются некоторым семейством вероятностных распределений.
Тогда задача кластеризации сводится к разделению смеси распределений по конечной выборке.
### EM-алгоритм
### Метод k-средних
### Кластеризация с частичным обучением