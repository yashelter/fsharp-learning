---
title : f#
---
### Бонусная база /0
+ [Лямбда-исчисление](https://neerc.ifmo.ru/wiki/index.php?title=Лямбда-исчисление#.D0.9B.D1.8F.D0.BC.D0.B1.D0.B4.D0.B0-.D0.B8.D1.81.D1.87.D0.B8.D1.81.D0.BB.D0.B5.D0.BD.D0.B8.D0.B5), используется в основе функциональных языков?

### Основы F# /1
F# создан Microsoft
- Functional first programming
- Взаимодействие с .NET (C#)
- Разрабатывать можно практически всё

Выводы:
+ Функциональное программирование - стиль, который ближе к математике
+ Функциональные программы проще распараллеливать
+ В чисто функциональных программах нет присваивания и изменения переменных, а значит и меньше побочных эффектов.  

Простые примеры:

Умножение на 2
```fsharp
let twice x = x * 2
twice 3
```

Второй вариант того же самого
```fsharp
(fun x -> x * 2) 3
```

Данные варианты эквиваленты:
```fsharp
let f x = x * 2
let f = (fun x -> x * 2)
```

Решение квадратных уравнений
```fsharp
let solve (a, b, c) =
    let d = b * b - 4. * a * c
    let x1 = (-b - sqrt (d)) / (2. * a)
    let x2 = (-b + sqrt (d)) / (2. * a)
    (x1, x2)

solve <| (1., 2., 1.)
solve <| (1., 2., -3.)
```

В функциональном программировании функции нескольких аргументов сводятся к функции одного аргумента. 

Как работает несколько аргументов (Карирование):

```fsharp
let plus x y = x + y;
let pre = (plus 1)
pre 2
```
Тут наглядно видно что технически мы не применяем аргументы, а сначала создаём функцию которая подставляет на место первого 1, а затем можно подставить другое число. Те можно понять как частичное выполнение функций, или подстановку в макрос. Также можно представить excel таблицу, где формулы к ячейкам применяются аналогично.

```fsharp
// если представить как лямбда выраждение становится очевидным как работает
let plus = fun x -> (fun y -> x + y)
```
> Если в функцию передавать один аргумент, то она уже не карируется, а выполняется, а кортеж является одним элементом

Итого:
- В функциональном программировании функции нескольких аргументов сводятся к функции одного аргумента
- Каррированая форма позволяет частичное применение функции

Условный оператор:

```fsharp
let max x y = if x > y then x else y

// если значение не возвращается (unit), то можно опустить else
let greater x y = if x > y then printfn "Truth"
```

Передача функции в качестве параметра
```fsharp
let table f =
    printfn " X     f(X)"
    for i = 0 to 10 do
        let x = 1./float(i)
        printfn "%7.4f    %7.4f" x(f, x)
```

Ещё интересные пример:

```fsharp
let min a b = if a > b then b else a
let max a b = if a > b then a else b

let triple f a b c = f (f a b) c
triple (+) 1 2 3 // 6
triple min 8 9 3 // 3
```
##### Выводы по главе \1:
- Функциональные программы очень лаконичны. В том числе из-за вывода типов.
- Все переменные - неизменяемые.
- Есть две базовые операции - определение функции и её применение. На их основе всё построено.
- Функции можно легко передавать в качестве аргументов в другие функции (функции высших порядков).
- Функции нескольких аргументов сводятся к функциям одного аргумента с помощью каррирования.
- В нем нет изменяемых переменных, поэтому нет побочных эффектов проще распараллеливать код
- Основная мощь - возможность композиции функций, работа с функциями как с данными
- Краткая запись (вывод типов)

### Функции и списки /2

Сначала введём основные понятия (выдадим базы)

**Абстракция** - создание функциональной константы из выражения
```fsharp
// тут нам по сути не важно как мы назовём x, а это лишь идея
// что это функция которая прибавит 1
fun x -> x + 1 
```
**Аппликация** - вызов функции с некоторыми данными
```fsharp
sin 0.5
```

**Редукция** — упрощение выражений в ходе вычислений (подстановка)

Редукция аппликации абстракции называется Beta-редукцией (вместо x подставить что-то)

Редукция специальных функций называется Delta-редукцией (замена функции)

В F# энергичный (аппликативный порядок редукции) 
При том, аппликативная редукция не всегда приводит к результату, а может приводить к зацикливанию (подробнее где-то в теории, которую не осветили)
```fsharp
(fun x -> x + 1) (2*3)// мы посчитаем сначала x=6
// в нормальном мы сначала подставим x=2*3, 
// а лишь в конце будем всё вычислять
```

**Комбинатор** — базовая функциональная единица, выражение без свободных переменных

Композиция:

```fsharp
// как они внутри
let (>>) f g = fun x -> g(f x)
let (<<) f g = fun x -> f(g x)

let (|>) x f = f x
let (>|) f x = f x
```

Пример

```fsharp
// все эквивалентны 
let f x = 2 * x + 1
let f = fun x -> 2 * x + 1

let f = fun x -> x |> (*) 2 |> (+) 1
let f = (*) 2 >> (+) 1
```

Система типов параметрический полиморфизм.

```fsharp
// прямая сумма int + string
type PesonAge =
    | Exact of int
    | Desc of string

let student = Exact(12)
let name = Desc("Ivan")

```

Option тип возвращает либо значение, если оно получено, либо None. Другими словами, это тип, который в зависимости от условий может принимать различные подтипы.
```fsharp
// уже в языке
type 'T option =
    | Some of 'T
    | None

// эквивалентно
type option<'T> =
    | Some of 'T
    | None

```
Удобно создавать свои типы, что бы обрабатывать выходные значения функций.
[Пример с квадратными уравнениями](https://youtu.be/CYBqdOelIWk?list=PL6XUtJhtlpPM3-1zyn5Ks6n7UB6gs38RS&t=615) 
```fsharp
type SolveResult =
    | None
    | Linear of float
    | Quadratic of float * float

let solve a b c =
    let D = b * b - 4. * a * c
    if a = 0. then
        if b = 0. then None 
        else Linear(-c / b)
    else if D < 0. then
        None
    else
        Quadratic(((-b + sqrt (D)) / (2. * a), (-b - sqrt (D)) / (2. * a)))

let x = solve 1. 2. -3.

```

Формальные выводы:
- Статическая типизация лучше динамической, т.к. позволяет дополнительно проверить правильность программы на этапе компиляции.
- В функциональных языках богатая система типов, включая функц. типы, полиморфизм и дизьюнктное объединение
- Система типов позволяет хорошо моделировать предметную область.

Сопоставление с образцом (например делается с помощью оператора match (что-то) with (список того-то)):

```fsharp
let print r =
    match r with
        | None -> printfn "No Solutions"
        | Linear(x) -> printfn "x=%f" x
        | Quadratic(x1, x2) -> printfn "x1=%f, x2=%f" x1 x2

print x // или можно так solve 1. 2. -3. |> print 
```

Следует обратить внимание в примере выше, на параметры в скобках, те мы по факту явно указываем чего ожидаем от `r`

Однако помимо типа, можно указать и условие, а также выбрать остальные результаты (else) с помощью `_`
```fsharp
let print r =
    match r with
        | None -> printfn "No Solutions"
        | Linear(x) -> printfn "x=%f" x
        | Quadratic(x1, x2) when x1=x2-> printfn "x1=x2=%f" x1 // условие
        | Quadratic(x1, x2) -> printfn "x1=%f, x2=%f" x1 x2
        | _ -> printfn "Not expected?" // пример else

print x // или можно так solve 1. 2. -3. |> print 
```

Однако не обязательно явно использовать оператор match, можно сделать тоже самое с помощью function:
```fsharp
let print r = function
        | Linear(x) -> printfn "x=%f" x
        | Quadratic(x1, x2) when x1=x2-> printfn "x1=x2=%f" x1
        | Quadratic(x1, x2) -> printfn "x1=%f, x2=%f" x1 x2
        | _ -> printfn "Not expected?"

print x
```
От сюда возникает вопрос в отличии задания анонимных функций с помощью `function` и `fun`
fun vs. function
`fun`
+ Поддерживает несколько аргументов в каррированной форме: fun xу-> ...
+ Не поддерживает pattern matching function

`function`
+ Поддерживает только один аргумент (возможно, tuple)
+ Поддерживает pattern matching с несколькими вариантами описания

Пример сопоставления пар:
```fsharp
let order = function
    | (x1, x2) when x2>x1 -> (x2, x1)
    | _ as pair -> pair

order (2, 4)
```

```fsharp
let x1, x2 = 'u', 'v' // попарное присвоение 

// пример реализации встроенных fst и snd
// также работают через сопоставление с образцом
let fst (x1, x2) = x1 
let snd (x1, x2) = x2
```

И весь смысл pattern matching, что удобнее чем условный оператор, и часто применяется неявно

#### Рекурсия

Понятие рекурсии эквивалентно её понятию в императивных языках. И рекурсия считается плохой, если она не хвостовая, которая может быть преобразована компилятором в цикл.

Примеры рекурсии:

```fsharp
// обычная рекурсия
let rec factorial n = 
    if n=1 then 1
    else n*factorial (n-1)

factorial 5
```

```fsharp
// обычная с сопостовлением
let rec factorial = function
    | 1 -> 1
    | n -> n * factorial(n-1)


factorial 5
```

```fsharp
// пример хвостовой рекурсии
let rec factorial n acc = 
    // появлеяется acc - аккумулятор результата
    if n=1 then acc
    else factorial (n-1) (acc * n)

factorial 5 1
```


### Списки, алгебраические типы \3
В функциональных языках нет, и не может быть массивов (тк это область памяти, а у нас тут все переменные не изменяемые), поэтому вместо них используются списки

> Тут можно возразить, тк в F# всё же есть массивы `[||]`, но это императивный элемент, а не функциональный, так что считаем, что их нет

</br>
Список в f# похож на список prolog'a (Elem, (Tail)), где Tail продолжение списка (другой список, возможно пустой `Nil`)

Пример списка и функции поиска его длины
```fsharp
//type 't list = Nil | Cons of 't * 't list // можно так
type list<'t> = Nil | Cons of 't * list<'t> // или так

let empty = Nil
let otf = Cons(1, Cons(2, Cons(3, Nil)))

// при том такой тип решения самый хороший
let rec get_length (acc:int) = function
  | Nil -> acc
  | Cons(_, tail) -> get_length  (acc + 1) tail

get_length 0 otf  // 3

```

> Задание для отработки сортировка и вставка с сохранением порядка, если не забуду вставлю тут под спойлер


```fsharp
// Стандартное обозначение списка на F#
type 'a list = 
    | ([])
    | (::) of 'a * 'a list
```
```fsharp
// задание списка пример
let squares = [   for i in 1 .. 20 do
                    if i % 5 = 0 then
                        yield i]
```

Хвост списка можно отделять с помощью оператора `Head_Lst::Tail_Lst` (либо вставлять)


Свёртка (fold) в F# - это функция высшего порядка, которая применяет некоторую операцию к элементам списка, последовательно сворачивая его до одного значения. Эта операция может быть суммированием, умножением, конкатенацией или любой другой операцией, определенной пользователем.
```fsharp
val List.fold : ('State -> 'T -> 'State) -> 'State -> 'T list -> 'State
```

```fsharp
// fold example
let rec fold_ f i = function
    | [] -> i
    | h::t -> f (fold_ f i t) h
    | _ as list -> 
        let t = List.tail list
        let h = List.head list
        f (fold_ f i t) h


fold_ (+) 0 [1..100]
```

Аналогом является Reduce, которая по факту такая же, но без использования начального значения.

```fsharp
// reduce example
let rec reduce_ f = function
    | [] -> failwith "Error"
    | [x] -> x
    | h::t -> f (reduce_  f t) h

reduce_ (+) [1..100]
```

Надо стараться при обходе списков использовать хвостовую рекурсию(и вообще всегда), тк она не требует дополнительного расхода памяти и может быть оптимизирована компилятором (тк по факту не требуются возвраты к начальному контексту)

```fsharp
// пример такого метода fold
let lst = [1..100]

let fold f i lstt = 
    let rec fold' acc = function
        | [] -> acc
        | h::t -> fold' (f h acc) t
    fold' i lstt
    
let result = fold (+) 0 lst
printfn "%d" result
```

Конкатенация списков
```fsharp
let f x acc = x::acc

let filteredList = List.foldBack f [1..10] [11..20] 

printfn "%A" filteredList
```

##### Основные функции списков

| Функция | Действие|
|-------------|-------------|
|  `List.empty `  | Возвращает пустой список    |
| `List.isEmpty`  | Проверяет на пустоту    |
| `List.head `   | Первый элемент списка    |
| ` List.tail `   | Список без первого элемента    |
| ` List.length `   | Длина списка    |
| ` List.exist `   | Есть ли элемент, который пройдёт условие    |
| ` List.rev `   | Реверсирование списка   |
| ` List.zip `   | Принимает 2 списка одной длины, и делает список кортежей    |
| ` List.filter `   | Новый список по старому, где выполнено условие |
| `List.parition`   | Два списка из исходного. Первый, где функция `true`, второй где `false`    |
| ` List.map`   | Применяет функцию к каждому элементу списка   |
| ` List.mapi`   | Применяет функцию к каждому элементу списка, у функции как параметр есть индекс   |
| ` List.init `   | Задание списка c длиной и функцией от индекса    |

Способы задания списков
```fsharp
List.init 9 (fun x->2.*float(x))
[for x in 0..8 -> 2.*float (x)]
[1..100]
[1..2..100]
[ for x in 1..10 do if x%3=0 then yield x]
```
**Немного примеров:**

Пример реализации поиска среднего значения в списке
```fsharp
let lst = [1..100]

let f acc elem = (fst acc + elem, snd acc + 1)

let getAverage L = 
    let (n, cnt) = List.fold f (0, 0) L
    float n / float cnt

getAverage lst
```

Транспонирование матрицы (списка)
```fsharp
let rec slice = function
    | [] -> ([], [])
    | h::t ->
        let (u, v) = slice t
        (List.head h ::u, List.tail h ::v)

slice [[1;2;3];[4;5;6]]


let rec transpon = function
| []::_  -> []
| x  ->
    let (h, r) = slice x
    h::(transpon r)

transpon [[1;2;3];[4;5;6]]
```

Сортировка списка выбором:

```fsharp
let rec fmin = function
    | [] -> failwith "unexpected"
    | [x] -> (x, [])
    | h::t -> 
        let (h1, t1) = fmin t
        if h<h1 then (h, t)
        else (h1,h::t1)

let rec sort = function
    | [] -> []
    | lst ->
        let (elem, other) = (fmin lst)
        elem::(sort other)

sort [1;4;6;7;12;13452;23;23;3]
```

Задача на составление списка простых чисел

```fsharp
// решето эратосфена
let lst = [2..100]

let rec simple = function
    | []  -> []
    | h::t -> h::simple(List.filter (fun x -> x % h<>0) t)
    
simple lst
```
<details>
  <summary>Задача на поиск степеней двойки</summary>
  
  Во время тестирования решета, был обноружен интересный способ поиска списка степеней двойки
  
  ```fsharp
  let lst = [2..10000] // обязательно с 2х начало!

  let rec simple = function
    | []  -> []
    | h::t -> h::simple(List.filter (fun x -> x % h=0) t) // =
            
  simple lst
  ```
  Решение пожалуй не самое наверное очевидное, но его можно доказать
</details>

#### Чуть подробнее про fold и foldBack
```fsharp
// правая свёртка List.foldBack
let rec foldr f i = function
    | [] -> i
    | h::t -> f h (foldr f i t)

// левая свёртка List.fold
let rec foldl f i = function
    | [] -> i
    | h::t -> foldl f t (f i h)
```
Правая:
+ Обходит список справа налево
+ Нехвостовая рекурсия

Левая:
+ Обходит список слева направо
+ Соответствует итерации
+ Хвостовая рекурсия
+ Более эффективная

### Деревья \4

Дерево общего вида типа T это - 
+ Элемент типа Т с присоединёнными к нему деревьями типа Т
+ Формально связанный граф без циклов
+ Пустого дерева общего вида не существует

Описание задания простого дерева
```fsharp
type 'T tree =
    | Leaf of 'T
    | Node of 'T * ('T tree list)

let tr = Node(1, 
            [Node(2,
                [Leaf(5)]);
            Node(3,[
                Leaf(6);
                Leaf(7)]);
            Leaf(4)])
```

Свёртка и отображение (map), достаточно тривиальны:
```fsharp
// отображение
let rec map f = function
    | Leaf(x)  -> Leaf(f, x)
    | Node(x, l) ->
        Node (f x,  List.map (map f ) l)

// свёртка
let rec fold f i = function
    | Leaf(x) -> f i x
    | Node(x, l) -> 
        List.fold (fold f) (f i x) l
```

Абстрактное синтаксическое дерево. Это максимально похоже на задание курсовой 
```fsharp
//  самое интересное что оно работает!
type 't Expr =
    | Add of 't Expr * 't Expr
    | Sub of Expr<'t> * Expr<'t>
    | Neg of 't Expr
    | Value of 't

let rec compute = function
    | Add(x,y) -> compute x + compute y
    | Sub(x,y) -> compute x - compute y
    | Neg(x) -> - compute x 
    | Value(x) -> x

compute(Add(Neg(Value(5)), Sub(Value(5),Value(1)))) // -5 + 5 - 1 = -1
```

Также существуют двоичные деревья. Любое дерево общего вида может быть преобразовано к двоичному. В [4.2 можно подробнее ознакомиться](https://www.youtube.com/watch?v=tyRXyp1C3Ds&list=PL6XUtJhtlpPM3-1zyn5Ks6n7UB6gs38RS&index=21)

```fsharp
// here maybe approach bin-trees
```

Далее упомяну что в F# можно создать BST, и был разобран его код, а также вставка в него. На примере показал как его использовать для поиска в по тексту [4.3](https://www.youtube.com/watch?v=i84485v0bQM&list=PL6XUtJhtlpPM3-1zyn5Ks6n7UB6gs38RS&index=22). Так как код большой(хотя не очень сложный), и запоминать не очень хочется, как и переписывать, просто оставлю ссылку, и понадеюсь что не пригодится. А также всякие деревья уже есть в .NET, в частости есть словарь `XD`, так что как будто можно пропустить.

#### Структуры продолжение 

Реализация наивной очереди

```fsharp
type 'a sillyqueue = 'a list

let rec put x = function
    | [] -> [x]
    | h::t -> h::(put x t)

let take = function
    | [] -> failwith "empty"
    | h::t -> (h, t)

let emptyq = []
```

Умная очередь - храним два списка, добавляем во второй, берём из первого, когда элементы в первом заканчиваются, перестраиваем его, копированием обратного второго:
```fsharp
type 'a queue = 'a list * 'a list

let tail (L,R) = 
    match L with
    | [x] -> (List.rev R, [])
    | h::t -> (t, R)

let head (h::_,_) = hash

let put x (L, R) =
    match L with
    | [] -> ([x], R)
    | _ -> (L, x::R)

let empty = ([],[])
```

Zippers - аналог двунаправленных списков, идейно похоже на умную очередь.
```fsharp
type 't ZipperList = 't list * 't list

let zip l :ZipperList<'t> = ([], l)
let unzip (l,r) = (List.rev l) @ r

let right (l, h::t) = (h::l, t)
let left (h::t, r) = (t, h::r)
let update x (l, h::r) = (l, x::r)
let insert x (l,r) = (l, x::r)

zip [1..3] |> right |> update 0 |> unzip
```

Zipper для дерева

Эвристики:
+ текущий элемент определяется путём до вершины
+ чтобы полностью определить дерево с текущим элементом, нужно хранить текущее поддерево + путь до него со всеми элементами для восстановления дерева
    - идя направо запоминаем левое поддерево
    - идя налево - правое
+ каждый шаг выворачивает дерево


```fsharp
type 'T tree =
    | Leaf of 'T
    | Node of 'T * 'T tree * 'T tree

let sample = Node ('+',  
    Node('*', Leaf('1'), Leaf('2')), 
    Node('*', Node('+', Leaf('3'), Leaf('4')), Leaf('5')))

type 't crumb = 
    | Left of 't * 't tree 
    | Right of 't * 't tree

type 't  TreeZipper = 't tree * 't crumb list

let zip t : 't TreeZipper = (t,[])

let left (Node(x,l,r), path) = (l, Left(x,r)::path)
let right (Node(x,l,r), path) = (r, Right(x, l)::path)

let update value (current, path)  =
    match current with 
    | Node(x, left, right) -> (Node(value, left, right), path)
    | Leaf(x) -> (Leaf(value), path)

let up (t, p) =
    match p with
    | Left (x,r)::xs -> (Node(x,t,r), xs)
    | Right (x, l)::xs -> (Node(x,l,t), xs)

zip sample |> right |> left |> update '0' |> up |> up
```

Важное преимущество : Zipper'ы позволяют модифицировать текущий, не перестраивая всей структуры

### Приёмы функционального программирования \5

Замыкание - совокупность функции и контекста: значений всех свободных переменных входящих в эту функцию, зафиксированных в момент определения функции

```fsharp
// вот такой пример для понимания
let x = 1
let f z = x+z
f 1

let x = 2 // но если скомпилить целиком ошибка
f 1
```

И в f# есть динамическое связывание
```fsharp
// новое старое слово
let mutable x = 1 // теперь тут ссылка
let f z = x+z
f 1

x <- 2 // ! внимание на изменение !
f 1
```
+ Замыкание это внутренние и внешние переменные
+ Замыкания позволяют сохранять внутри функции некоторое состояние
+ Изменяемые переменные - штука для изменения состояния (императивный прикол, не функциональный уже считается)

#### Примеры задач с mutable и замыканием
Подсчёт суммы списка
```fsharp
// с mutable
let lst = [1..100]

let mutable res = 0

let rec f = function
    | [] -> None
    | h::t -> 
        res <- (res + h) 
        f t
f lst
printf "%d" res
```

Генератор простых чисел
```fsharp
let lst = [2..100]

let rec simple = function
    | [] -> []
    | h::t -> h::simple(List.filter (fun x -> x % h <>0) t)

let mutable simpels = simple lst

let get_simple = function 
    | [] -> 0
    | h::t ->
        simpels <- t
        h

let f = get_simple simpels
```

Генераторы - штука, которая (генерирует) последовательность (возможно бесконечную). Те это иное представление списка, хотя при этом абстракция. Это пример ленивых (отложенных вычислений)
```fsharp
// генератор общего вида
let new_generator fgen init =
    let mutable x = init
    fun () ->
        x <- fgen x;x

// модификация генератора
let map f gen = 
    fun() -> f (gen())

// фильтр для генератора
let rec filter cond gen =
    fun() ->
        let x = gen()
        if cond x then x
        else (filter cond gen) ()

// взятие n элементов у генератора
let rec take n gen =
    if n=0 then[]
    else gen()::take(n-1) gen


let fibgen = new_generator (fun (u, v) -> (u+v,u)) (1, 1)

// example
fibgen |> map fst |> take 10
```

#### Последовательности
Последовательности - тип данных похожий на список и генератор. (LazyList .Net (но с кэшированием результатов)) Те элементы могут не вычисляться до 1го обращения

```fsharp
// методы задания
let squares = Seq.initInfinite(fun n -> n*n)
let squares10 = Seq.init 10 (fun n ->n*n)


seq {1..100}
seq {for x in 1..10 -> x*x}
seq {for x in 1..10 do
        if x%3=0 then yield x*x}

```


```fsharp
// do 5.3 lesson (13:00)

```
```fsharp
// способы задать факториал, это ответ на 1 из

let fact n =  [1..n] |> List.reduce (*)

let fact n = seq {1..n} |> Seq.reduce (*)

// (..) оператор задания последовательности
let fact  = (..) 1 >> Seq.reduce (*) 

let fact n = Seq.initInfinite ((+) 1) |> Seq.scan (*) 1 |> Seq.nth n
```

Корекурсия - определение рекуррентной бесконечной последовательности структурно эквивалентное рекурсии. Обычно пораждает бесконечные последовательности

```fsharp
// пример определения корекурсии
let rec ones = seq{
    yield 1;
    yield! ones;
}

ones |> Seq.take 10 //1 1 1..
```

```fsharp
// ещё пример определения корекурсии
let rec nat = seq{
    yield 0
    yield! 
        Seq.map ((+) 1) nat
}

nat |> Seq.take 10 // 0..10
```

</br>

Следующий прикол - продолжения - штука позволяющая сводить нелинейную рекурсию к хвостовой

Как мне кажется не очень обязательная штука. Она также применима к деревьям [смотреть 5.5](https://www.youtube.com/watch?v=CKeGcqg0pI8&list=PL6XUtJhtlpPM3-1zyn5Ks6n7UB6gs38RS&index=28)
```fsharp
// пример продолжений для длины списка
let len L =
    let rec len' cont = function
        | [] -> cont 0
        | h::t -> len' (fun x -> cont(x+1)) t
    len' (fun x->x) L 
// те до конца вычислений мы собираем длинную конструкцию в виде функций
// а как дошли вычисляем его
// мы обошлись без аккумулятора и получили хвостовую рекурсию
```

Далее, Мемоизация - запоминание вычесленных значений, для оптимизации. Применимо когда каждому входному -> единственное выходное всегда. 

Мы жертвуем памятью на скорость. Данный приём иногда применяется в добрых алгоритмах (в том числе на олл проге).
```fsharp
// вот общая идея
open System.Collections.Generic

let memoize (f: 'a -> 'b) = 
    let t = new Dictionary<'a,'b>() // c# hi
    fun n -> 
        if t.ContainsKey(n) then t.[n]
        else let res = f n 
             t.Add(n, res) 
             res


let rec fibFast = 
    memoize(
    fun n -> if n < 2 then 1
             else fibFast(n-1) + fibFast(n-2))
```

### Монады и монадические выражения \6

По началу может быть не понятно, но постараюсь описать, чтобы к концу раздела стало понятно `uwu`
Сперва стоит начать с примеров:

```fsharp
// общая идея
open System

let read() =
    printf ">"
    let s = Console.ReadLine()
    try
        Some((int)s)
    with _ -> None

let bind a f =
    if a=None then None
    else f (a.Value)

bind (read()) (fun x -> bind (read()) (fun y -> Some(x+y)))
```

bind берёт значение a и передаёт его на вход функции f, при этом происходит дополнительная обработка.

```fsharp
// запись с помощью опратора
let (>>=) a f =
    if a=None then None
    else f (a.Value)

read() >>= fun x -> read() >>= fun y -> Some(x + y)
```

Другой пример:

```fsharp
let inv x =
    if x <> 0. then Some(1./ x)
    else None

let mmap f = fun x -> Some(f x) // вспомогательная функция для оборачивания в Optional
read() >>= nmap float >>= inv
```
Итак монада:
+ Монадический тип `M<T>`
+ Операция return: `T -> M<T>`
+ Операция `>>=` :` M<A> -> (B -> M<B>) -> M<B>`

Монадические законы:
+ `return x >>= f` == `f x` : левая единица
+ `m >>= return `== `m`  : правая единица
+ `(m >>=f ) >>= g` == `m >>= (fun x -> f x >>= g)` : ассоциативность

Монадические выражения

```fsharp
type NondetBuilder() = 
    member b.Return x = [x]
    member b.Bind(mA, f) = List.collect f mA

let nondet = new NondetBuilder()

let r = nondet {
        let! vas = [ 14; 15 ]
        let! pet = [ 2 * vas; 3 * vas ]
        let lena = pet + 1
        return lena
    }

```
`let!` показывает что должен сработать оператор Bind, что можно понимать как распаковку объекта

+ Монада (нестрого) - некоторый обрамляющий тип + набор специальных операций.
+ Монада позволяет описывать в явном виде последовательность некоторых операций, семантику выполнения которых мы можем менять.

### Бонусы \7
> Тут чисто тезисно, пока не понадобится в использовании, так как на экзамен вряд ли будет вынесено, но чисто идейно может пригодиться.

Существуют асинхронные агенты. Если подробней интресно то [вот урок](https://www.youtube.com/watch?v=1I9kNSC1a-0&list=PL6XUtJhtlpPM3-1zyn5Ks6n7UB6gs38RS&index=37), пока не вижу смысл в углублении таком

```fsharp
// there will be code (maybe )
// awaitig MR :)

```

Активные шаблоны

```fsharp
// пример функции вычисляющей эмоциональную окраску текста
type System.String with 
    member x.MySplit() = x.Split([|'"';..|])

let mood (s:string) = 
    s.MySplit() |> Array.fold(fun z w -> 
        if Set.contains w poset then z+1
        elif Set.contains w neset then z-1
        else z) 0
```

```fsharp
// пример активного шаблона
let (|Positive|Negative|Neutral|) (w:string) = 
    let s = w.ToLower()
    if Set.contains s poset then Positive
    elif Set.contains s neset then Negative
    else Neutral

let mood (s:string) =
    s.MySplit()
    |> Array.fold(fun z w ->
        match w with
            | Positive -> z + 1
            | Negative -> z-1
            | Neutral -> z) 0
```
Далее упомяну, что шаблоны могут использоваться в регулярных выражениях, если вдруг понадобится есть в лекции 7.2. Ну и есть инструмент для графиков FsharpChart
