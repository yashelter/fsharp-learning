// For more information see https://aka.ms/fsharp-console-apps
printfn "Hello from F#"

let twice x = x * 2
twice 3


let solve (a, b, c) =
    let d = b * b - 4. * a * c
    let x1 = (-b - sqrt (d)) / (2. * a)
    let x2 = (-b + sqrt (d)) / (2. * a)
    (x1, x2)

solve <| (1., 2., 1.)
solve <| (1., 2., -3.)

(fun x -> x * 2) 3


let plus x y = x + y
let pre = (plus 1)
pre 2

let greater x y =
    if x > y then
        printfn "Truth"

greater 5 4

let min a b = if a > b then b else a
let max a b = if a > b then a else b

let triple f (a: int) (b: int) (c: int) = f (f a b) c
triple (+) 1 2 3
triple min 8 9 3


let (>>) f g = fun x -> g (f x)
let (<<) f g = fun x -> f (g x)

let (|>) x f = f x
let (>|) f x = f x

let f x = 2 * x + 1
let f = fun x -> 2 * x + 1

let f = fun x -> x |> (*) 2 |> (+) 1
let f = (*) 2 >> (+) 1
f 3


type PesonAge =
    | Exact of int
    | Desc of string

let student = Exact(12)
let name = Desc("Ivan")

let rec fib n =
    match n with
    | 0
    | 1 -> n
    | n -> fib (n - 1) + fib (n - 2)

fib 10

type SolveResult =
    | None
    | Linear of float
    | Quadratic of float * float

let solve a b c =
    let D = b * b - 4. * a * c

    if a = 0. then
        if b = 0. then None else Linear(-c / b)
    else if D < 0. then
        None
    else
        Quadratic(((-b + sqrt (D)) / (2. * a), (-b - sqrt (D)) / (2. * a)))

let x = solve 1. 2. -3.

let print =
    function
    | None -> printfn "No Solutions"
    | Linear(x) -> printfn "x=%f" x
    | Quadratic(x1, x2) when x1 = x2 -> printfn "x1=x2=%f" x1 // условие
    | Quadratic(x1, x2) -> printfn "x1=%f, x2=%f" x1 x2
    | _ -> printfn "Not expected?" // пример else

print x
