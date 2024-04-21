let epsi = 1e-5


let rec dichotomy (f: float -> float) a b =
    let d = (a + b) * 0.5

    if abs (a - b) < epsi then d
    else if (f a * f d > 0) then dichotomy f d b
    else dichotomy f a d

let rec iterations phi x0 =
    let x1 = phi x0
    if abs (x1 - x0) < epsi then x1 else iterations phi x1


let rec neython f f' x0 =
    let phi x = x - f x / f' x
    iterations phi x0

let f1 x =
    0.6 * (System.Math.Pow(3, x)) - (2.3 * x) - 3.

let f2 x = x * x - System.Math.Log(1. + x) - 3.

let f3 x =
    2. * x * System.Math.Sin(x) - System.Math.Cos(x)


let f1' x =
    System.Math.Log(3) * (System.Math.Pow(3, x + 1.)) * 0.2 - 2.3

let f2' x = 2. * x - (1. / (1. + x))

let f3' x =
    3. * System.Math.Sin(x) + 2. * x * System.Math.Cos(x)

// g(x) = x - f(x) / f'(x)

let g1 x =
    (-3. + 0.6 * System.Math.Pow(3, x)) / 2.3 // данное уравнение (1) нельзя решить методом итераций из за не выполнения условия сходимости метода итераций

let g2 x =
    System.Math.Sqrt(System.Math.Log(1. + x) + 3.)

let g3 x = (((1. / (System.Math.Tan(x))) / 2.))



let main =
    printfn "|======================================================================|"
    printfn "|      Dichotomy               Iterations                 Neython      |"
    printfn "|======================================================================|"

    printfn "| %3.18f\t Imposible   \t\t  %3.18f |" (dichotomy f1 2. 3.) (neython f1 f1' 2.)
    printfn "| %3.18f\t %13.18f\t  %3.18f |" (dichotomy f2 2. 3.) (iterations g2 2.) (neython f2 f2' 2)
    printfn "| %3.18f\t %3.18f\t  %3.18f |" (dichotomy f3 0.4 1.) (iterations g3 0.4) (neython f3 f3' 0.4)

main
