let epsi = 1e-4


let rec dichotomy (f: float -> float) a b =
    let d = (a + b) * 0.5

    if abs (a - b) < epsi then d
    else if (f a * f d > 0) then dichotomy f d b
    else dichotomy f a d

let rec iterations phi x0 =
    let x1 = phi x0
    if abs (x1 - x0) < epsi then x1 else iterations phi x1


let rec newthon f f' x0 =
    let phi x = x - f x / f' x
    iterations phi x0

// Решите 3 уравнения (начиная со своего номера варианта) с использованием 3-х методов
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
let phi1 x =
    (-3. + 0.6 * System.Math.Pow(3, x)) / 2.3

let phi2 x =
    System.Math.Sqrt(System.Math.Log(1. + x) + 3.)

let phi3 x = (((1. / (System.Math.Tan(x))) / 2.))


//(dichotomy f1 2. 3.) (iterations phi1 -1.) (newthon f1 f1' 2.)
(newthon f2 f2' 2.5)
//(dichotomy f3 0.4 1.) (iterations phi3 0.4) (newthon f3 f3' 0.4)

let main =
    printfn "%10.10f  %10.10f  %10.10f" (dichotomy f1 2. 3.) (iterations phi1 -1.) (newthon f1 f1' 2.)
    printfn "%10.10f  %10.10f  %10.10f" (dichotomy f2 2. 3.) (iterations phi2 2.) (newthon f2 f2' 2)
    printfn "%10.10f  %10.10f  %10.10f" (dichotomy f3 0.4 1.) (iterations phi3 0.4) (newthon f3 f3' 0.4)

main
