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
