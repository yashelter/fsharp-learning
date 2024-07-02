# О репозитории и авторах 
Это конспект составленный [мной](https://github.com/yashelter) во время прохождения институтского курса функциононального программирования. Также тут можно найти работу <i><b>[по компилятору на F#](/fp-compiler-lab-starlight-sky/)</i></b>, к которому также приложили руку [ZBelka100](https://github.com/ZBelka100) и [MaximusPokeZ](https://github.com/MaximusPokeZ).
Если у вас есть вопросы или предложения можете смело писать их в issues или мне лично в [![Telegram](https://img.shields.io/badge/telegram-%231DA1F2.svg?&logo=telegram)](https://t.me/yashelter)

Если вам помог мой скромный труд, то пожалуйста поставьте звёздочку, мне будет приятно
![Alt Text](/staff/yu.gif)
## Небольшое оглавление: 
- [О репозитории и авторах](#о-репозитории-и-авторах)
  - [Небольшое оглавление:](#небольшое-оглавление)
    - [Полезные ссылки /0](#полезные-ссылки-0)
    - [Основы F# /1](#основы-f-1)
        - [Выводы по главе \\1:](#выводы-по-главе-1)
    - [Функции, и их применение /2](#функции-и-их-применение-2)
      - [fun vs. function:](#fun-vs-function)
      - [Рекурсия](#рекурсия)
    - [Списки, алгебраические типы \\3](#списки-алгебраические-типы-3)
        - [Некоторые функции списков](#некоторые-функции-списков)
      - [Немного примеров](#немного-примеров)
      - [Чуть подробнее про fold и foldBack](#чуть-подробнее-про-fold-и-foldback)
    - [Различные структуры \\4](#различные-структуры-4)
      - [Дерево общего вида](#дерево-общего-вида)
      - [Очереди](#очереди)
      - [Zippers](#zippers)
    - [Приёмы функционального программирования \\5](#приёмы-функционального-программирования-5)
      - [Замыкания](#замыкания)
      - [Примеры задач с mutable и замыканием](#примеры-задач-с-mutable-и-замыканием)
      - [Генераторы](#генераторы)
      - [Последовательности](#последовательности)
      - [Продолжения](#продолжения)
      - [Мемоизация](#мемоизация)
    - [Монады и монадические выражения \\6](#монады-и-монадические-выражения-6)
      - [Сперва примеры](#сперва-примеры)
      - [Определение монад](#определение-монад)
      - [Монадические законы](#монадические-законы)
      - [Монадические выражения](#монадические-выражения)
      - [Теория категорий, типизация и функциональное программирование](#теория-категорий-типизация-и-функциональное-программирование)
        - [Категории](#категории)
      - [Функторы](#функторы)
        - [Категориальное определение пары](#категориальное-определение-пары)
      - [Снова Монады](#снова-монады)
      - [Оптика](#оптика)

### Полезные ссылки /0
+ [Лямбда-исчисление](https://neerc.ifmo.ru/wiki/index.php?title=Лямбда-исчисление#.D0.9B.D1.8F.D0.BC.D0.B1.D0.B4.D0.B0-.D0.B8.D1.81.D1.87.D0.B8.D1.81.D0.BB.D0.B5.D0.BD.D0.B8.D0.B5)
+ [Применение функторов и монад](https://habr.com/ru/articles/686768/)
+ [Документация F#](https://learn.microsoft.com/en-us/dotnet/fsharp/)


### Основы F# /1

Что такое F#:
- Functional first programming
- Взаимодействие с .NET (C#)
- Разрабатывать можно практически всё

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
// если представить как лямбда выраждение становится очевидным как работает карирование
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

### Функции, и их применение /2

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

<br>
В F# энергичный (аппликативный порядок редукции) 
При том, аппликативная редукция не всегда приводит к результату, а может приводить к зацикливанию.

```fsharp
(fun x -> x * 2) (2*3) // мы посчитаем сначала x=6
// в нормальном (ленивом ((не F#)))  мы сначала подставим x=2*3, 
// а лишь в конце будем всё вычислять
```

**Комбинатор** — базовая функциональная единица, выражение без свободных переменных

```fsharp
// комбинаторы: 
let I x = x // единичный
let K x y = x // канцелятор

let S x y z = x z (y z) // распределитель
let flip f x y = f y x // смена порядка
```

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
//Комбинатор
let combine f g x = f (g x) // <<

let addOne x = x + 1
let double x = x * 2

let addOneThenDouble = combine double addOne

printfn "%d" (addOneThenDouble 3)

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
        | Quadratic(x1, x2) when x1=x2 -> printfn "x1=x2=%f" x1 // условие
        | Quadratic(x1, x2) -> printfn "x1=%f, x2=%f" x1 x2
        | _ -> printfn "Not expected?" // пример else

print x // или можно так solve 1. 2. -3. |> print 
```

Однако не обязательно явно использовать оператор match, можно сделать тоже самое с помощью function:
```fsharp
let print = function
        | Linear(x) -> printfn "x=%f" x
        | Quadratic(x1, x2) when x1=x2-> printfn "x1=x2=%f" x1
        | Quadratic(x1, x2) -> printfn "x1=%f, x2=%f" x1 x2
        | _ -> printfn "Not expected?"

print x
```
От сюда возникает вопрос в отличии задания анонимных функций с помощью `function` и `fun`

#### fun vs. function:

`fun`
+ Поддерживает несколько аргументов в каррированной форме: fun xу-> ...
+ Не поддерживает pattern matching function

`function`
+ Поддерживает только один аргумент (например tuple)
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

> Тут можно возразить, тк в F# всё же есть массивы `[||]`, но это *императивный* элемент, а не *функциональный*. 

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

// или
let a = [1..10]
let b = [11..2..180]

let rec concat lst1 = function
    |[] -> lst1
    |h::t -> concat (lst1@[h]) t
    
printfn "%A" (concat a b)

```

##### Некоторые функции списков

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
| `List.partition`   | Два списка из исходного. Первый, где функция `true`, второй где `false`    |
| ` List.map`   | Применяет функцию к каждому элементу списка   |
| ` List.mapi`   | Применяет функцию к каждому элементу списка, у функции как параметр есть индекс   |
| ` List.init `   | Задание списка c длиной и функцией от индекса    |
| ` List.allpairs `   | Декартового произведение двух списков   |

Способы задания списков
```fsharp
List.init 9 (fun x->2.*float(x))
[for x in 0..8 -> 2.*float (x)]
[1..100]
[1..2..100]
[ for x in 1..10 do if x%3=0 then yield x]
```
#### Немного примеров

Пример реализации вычисление среднего арифметического значения в списке
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

### Различные структуры \4

#### Дерево общего вида
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
    | Leaf(x) -> Leaf(f x)
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

Также существуют двоичные деревья. Любое дерево общего вида может быть преобразовано к двоичному

#### Очереди

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

#### Zippers
Zippers - аналог двунаправленных списков, где мы смортим на 1 элемент и имеем опрерации сдвигов.

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

![Котик](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)

### Приёмы функционального программирования \5

#### Замыкания
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

#### Генераторы
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
// способы задать факториал

let fact n =  [1..n] |> List.reduce (*)

let fact n = seq {1..n} |> Seq.reduce (*)

// (..) оператор задания последовательности
let fact  = (..) 1 >> Seq.reduce (*) 

let fact n = Seq.initInfinite ((+) 1) |> Seq.scan (*) 1 |> Seq.nth n
```

Корекурсия - определение рекуррентной бесконечной последовательности структурно эквивалентное рекурсии.

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
    yield! Seq.map ((+) 1) nat
}

nat |> Seq.take 10 // 0..10
```
#### Продолжения
Продолжения - штука позволяющая сводить нелинейную рекурсию к хвостовой

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
#### Мемоизация
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


По началу может быть не понятно, однако чем больше погружаешься, тем более понятно должно становиться, пока снова не станет непонятно `uwu`:)

#### Сперва примеры

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

#### Определение монад
+ Монадический тип `M<T>`
+ Операция return: `T -> M<T>`
+ Операция `>>=` :` M<A> -> (B -> M<B>) -> M<B>`

#### Монадические законы
+ `return x >>= f` == `f x` : левая единица
+ `m >>= return `== `m`  : правая единица
+ `(m >>=f ) >>= g` == `m >>= (fun x -> f x >>= g)` : ассоциативность

#### Монадические выражения

```fsharp
type NondetBuilder() = 
    member b.Return x = [x]
    member b.Bind(mA, f) = List.collect f mA

let nondet = new NondetBuilder()

let r = nondet {
        let! vas = [14; 15]
        let! pet = [2 * vas; 3 * vas]
        let lena = pet + 1
        return lena
    }
```

`let!` показывает что должен сработать оператор Bind, что можно понимать как распаковку объекта

+ Монада (нестрого) - некоторый обрамляющий тип + набор специальных операций.
+ Монада позволяет описывать в явном виде последовательность некоторых операций, семантику выполнения которых мы можем менять.


Хороший пример для понимания:
```fsharp
type ResultBuilder() =
    member _.Bind(x, f) = 
        match x with
        | Some(v) -> f v
        | None -> None
    member _.Return(v) = Some(v)

let result = ResultBuilder()

let computation = result {
    let! x = Some(2)
    do! if x > 5 then Some () else None
    let! y = Some(20)
    return x + y
}

printfn "%A" computation
```
Аналог:
```fsharp
type OptionBuilder() =
    member _.Return(x) = Some x
    member _.Bind(m, f) = 
        match m with
        | Some x -> f x
        | None -> None

let option = OptionBuilder()


let computation = 
    option.Bind(Some 10, 
        fun a -> option.Bind(Some 20, 
            fun b ->option.Return(a + b)))
```

`do!` - делает что то, а значение используется для проверки в цепочке выражения


Монада Writer - записывает логи по ходу выполнения
```fsharp
let add x y =
    writer {
        do! writer.Tell (sprintf "Log...%A %A" x y)
        return x + y
    }
```

```fsharp
// пример реализации
type Writer<'t> = 't*string

type WriterBuilder() = 
    member m.Run (t,s) = (t,s) 
    member m.Tell (s:string) = ((),s)
    member m.Bind(x,f) =
        let (v,s:string) = m. 
        Run x let (v',s') = m.Run (f v) 
        (v',s+s')
    member m.Return x = (x,"")

let writer = new WriterBuilder()
```
Монада ввода вывода - записывает логи по ходу выполнения
Монада состояния - позволяет сохранить состояние, имеет методы get, set
```fsharp
type State <'s, 't> = 's -> 't* 's
type StateBuilder() = 
    member m.Bind(x, f) = 
        (fun s0 -> 
        let a, s = x s0 
        f a s)
    member m.Return a = (fun s -> a, s)
    member m.Zero() = m.Return()

let state = new StateBuilder()

let getState = (fun s -> s, s)
let setState s = (fun _ -> (s), s)
```

Монада продолжений
```fsharp
type ContinuationBuilder() =
    member this.Bind (m, f) =
        fun c -> m (fun a -> f a c)
    member this.Return x = fun k -> k x

let cont = ContinuationBuilder()
```

Также монады можно использовать для параллейных вычислений и парсеров (Fparsec работает схоже (вспоминаем синтаксис))

#### Теория категорий, типизация и функциональное программирование
+ Взгляд на теорию функционального программирования на основе очень абстрактного математического понятия
+ Изучает взаимосвязь понятий без привязки к их внутренней структуре
+ Применяется в: 
    - Топологии
    - Теоретической физике
    - Информатике

##### Категории
Определение - это семейство объектов Ob(K), и семейство морфизмов ("стрелок") Mor(K)
- $\forall f : A \to B, g: B \to C, \exists g \circ f : A \to C$ (композиция)
- $\forall A \in \text{Ob}(K), \exists \text{Id}_A: A \to A$ (единичная стрелка)
- $\forall f, g, h \in \text{Mor}(K): (f \circ g) \circ h = f \circ (g \circ h)$ (ассоциативность)
- $\forall f : A \to B$ имеет место $f \circ \text{Id}_A = \text{Id}_B \circ f = f$ (свойство единицы)

Примеры категорий:
+ Категория из одного объекта
+ Граф пораждает категории
+ Отношение порядка пораждает категорию
+ Категория всех множеств Set (каждый объект множество, "стрелки" - функции)
+ Категория всех типов\функций ЯП (надо рассматривать возможную незавершимость функций)

**Двойственные категории** - такие категории, что "стрелки" повернули наоборот. Удобны для доказательств (по двойственности... ).
+ $Ob(K) = Ob(K_{duo})$
+ $\forall f : A \rightarrow B \in  Mor(K),$ $\exist g: B \rightarrow A \in Mor(K_{duo})$

Начальный и терминальный объект
- $A \in Ob(K)$ - начальный объект, если $\forall B \in Ob(K) \ \exists ! f : A \rightarrow B$ 
    + Из него идут стрелки во все другие объекты
- $A \in Ob(K)$ - терминальный объект, если $\forall B \in Ob(K) \ \exists ! f : B \rightarrow A$ 
    + из каждого объекта сюда приходит одна стрелка

Пример начальные и терминальные объекты в Set:
+ absurd: void $\rightarrow$ 't - начальный
+ unit: `let Unit _ = ()`

#### Функторы

- Функтор $F$ – это отображение двух категорий $K_1$ и $K_2$
- Функтор должен сохранять свойства композиции
- $F : Ob(K_1) \rightarrow Ob(K_2)$
- $F : Mor(K_1) \rightarrow Mor(K_2)$

Свойства:
- $F(Id_A) = Id_{F(A)}$
- Если $f : A \rightarrow B$, то $F \ f : F(A) \rightarrow F(B)$
- $F(f \circ g) = F(f) \circ F(g)$

<details>
<summary>Примеры некоторых функторов</summary>
Конструктор : `fmap: (a->b) -> (F a -> F b)`

Some: 
```fsharp
let fmap f = function 
    | None -> None
    | Some(x) -> Some(f x)
```

[]: 
```fsharp
List.map
```

Дерево типа t: 
```fsharp
Tree Map
```
</br>
</details>

</br>
Функтор отображающий категорию в саму в себя, называется <u><b>эндофунктором</u></b>

Функтор Some на F#:
```fsharp
let fmap f = function
    | None -> None
    | Some(x) -> Some(f x)

fmap ((+) 1) (Some 10)

let ($) f x = fmap f x // производит лифтинг?
(+) 1 $ Some(10) 

```
**Лифтинг** - подъём на уровень функтора `(5 -> 6 : Some(5) -> Some(6))`. Те повышение абстрактности.

**Аппликативные функторы** - функторы, для которых мы рассматриваем исходные фукнкции в пространстве функторного типа

```fsharp
let (<*>) f x = // example
    match f, x with
        | None, _ -> None
        | _, None -> None
        | Some(f), Some(x) -> Some(f x)

(Some (+)) <*> Some(1) <*> Some(2) // Some 3
```

`Some((+))` - можно записать как функцию `Pure(f)`, которая будет поднимать функции

```fsharp
let (<!>) a b =
// для другой функции на вход, вернёт список, где для каждого элемента применит входную функцию
    let funct = (fun f -> List.map f b) 
    let lst = List.map funct a // применим для списка a
    lst |> List.concat 

let res = [(+)1;(+)(-1)] <!> [1;2]
printfn "%A" res

let pure f = [f]
pure (+) <!> [-1;-2] <!> [3..4]
```

Задача. Надо расставить знаки(+,-,\*) между числами в списке, и посчитать все возможные значения. `[1;2] -> [1*2;1-2;1+2]`

```fsharp
let rec split = function
    | [x] -> []
    | h::t -> ([h], t)::(List.map (fun (a, b) -> (h::a, b)) (split t))

split([1..10])

// все варианты расстановки знаков между числами в списке
let rec values = function
    | [] -> []
    | [x] -> [x]
    | Lst -> 
        let variats = split Lst
        let res = List.map (fun (l, r) -> [(+);(-);(*)] <!> (values l) <!> (values r)) variants
        List.distinct (List.concat res)

```

##### Категориальное определение пары

- Произведение $A, B \in Ob(K)$ это:
  - Объект $C \in Ob(K)$
  - Пара стрелок $\text{fst}: C \rightarrow A$, $\text{snd}: C \rightarrow B$ (проекции)

- При этом $\forall D \in Ob(K)$ и $g_1 : C \rightarrow A$, $g_2 : C \rightarrow B$ $\exists ! f : D \rightarrow C$

+ В категории Set произведение – это $A \times B$

*Произведение определяется с точностью до изоморфизма*

Алгебраические типы данных можно определять на основе произведения и суммы. 
Например
|Значение|Тип|
|---|---|
|1|()|
|a + b|Either a b = Left a | Right b|
|a * b|(a, b)|
|2 = 1 + 1|type Bool = True | False|
|1 + a|type 't option = None | Some 't|
|0 * a|(Void, 'a)|
|a \* (b + c) = a \* b +a \* c| ('a, Left 'b \| Right 'c) = Left ('a, 'b) \| Right('a, 'c)|
|1 + 't * x|list<'t> = Nil \| Const 't * list<'t>|

#### Снова Монады
>Монады это моноид в категории эндофункторов

Монады позволяют описать цепочку операций, где что то может пойти не так

```fsharp
let good x = Some(x+1)
let bad x = None

let (>>=) x f = 
    match x with
        | Some(z) -> f z
        | None -> None

Some(1) >>= good // Some(2)
Some(1) >>= bad  >>= good // None
```

Т.Е. это такая штука, которая за нас обработает данные, на случай если они станут плохими в какой то момент, убирая большую вложенность блоков `if-else`
Математически называется категория Клейси(`>=>`(fish operator)), отличие с bind(`>>=`), что на вход уже подаётся обёрнутое значение.

Сравнение 
|Название|Функция|
|---|---|
|Функтор|`fmap (a->b) -> F a -> F b`|
|Аппликативный функтор|`<*> : F (a -> b) -> F a -> F b` <br>`pure f : (a -> b) -> F(a -> b)`|
|Монада|`>>= : M a -> (a -> M b) -> M b` <br>`>=> : (a -> M b) -> (b -> M c) -> (a -> M c)` <br> `return: a -> M a`|

</br>

#### Оптика
**Оптика** - набор функциональных абстракций позволяющих работать со сложными структурами данных

**Линзы** решают задачу выделения конкретного места в какой-то структуре данных

Примеры линз с применением
```fsharp
type Content = 
    | String of string 
    | Body of Content list
    | Collection of Content list
    | Paragraph of Content
    | Header of int * Content
    | Bold of Content
    | ItemList of Content
    | TableRow of Content list
    | Table of Content list

let doc = Body([
    Header(1, String "title");
    Paragraph(String("This is introduction"));
    Table([
        TableRow([String("Item 1"); String("$1")]);
        TableRow([String("Item 2"); String("$2")]);
        ])
])

// пример для списка
type Get_function<'t, 'a> = ('t -> 'a)
type Update_function<'t, 'a> = ('t -> 'a -> 't) 
type Lense<'t, 'a> =  Get_function<'t, 'a> * Update_function<'t, 'a>


let get_func lense lst = fst lense lst
let update_func lense lst = snd lense lst 

let list_lense n = Lense(
    (fun (lst:List<'t>) -> lst.[n]),// get
    (fun lst new_element -> List.mapi (fun i t -> if i=n then new_element else t) lst) // update
)

let lst = [0..10]
get_func (list_lense 4) lst

let lst' = update_func (list_lense 2) lst -5
get_func (list_lense 2) lst'

let inline (.>) (xget, xset) (yget, yset) = 
   (xget >> yget), 
   (fun matrix element -> xset matrix (yset (xget matrix) element))


let mtx = [[1; 2; 3; 4; 5]; [6; 7; 8; 9; 10]]
get_func (list_lense 0 .> list_lense 0) [[1; 2; 3; 4; 5]; [6; 7; 8; 9; 10]]
update_func (list_lense 1 .> list_lense 1) [[1; 2; 3; 4; 5]; [6; 7; 8; 9; 10]] 0


let body = Lense(
    (function Body(x) -> x),
    (fun _ x -> Body(x))
)
let par = Lense(
    (function Paragraph(x) -> x),
    (fun _ x -> Paragraph(x))
)
let table = Lense(
    (function Table(x) -> x),
    (fun _ x -> Table(x))
)
let tablerow = Lense(
    (function TableRow(x) -> x),
    (fun _ x -> TableRow(x))
)

get_func (body .> list_lense 2 .> table .> list_lense 1 .> tablerow .> list_lense 1) doc
```

![Гладить](/staff/cute.gif)
