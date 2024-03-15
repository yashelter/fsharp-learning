let epsilon = 1e-8

let rec iter f acc i b =
    if i <= b then f (iter f acc (i + 1) b) i else acc

let power x n = iter (fun acc i -> acc * x) 1. 1 n

let funct x = 1. / (2. * float x - 5.)

let funct_n x i =
    -((power 2. (i - 1)) * (power x (i - 1)) / (power 5. i))

let funct_prev x sn = (sn * 2. * x - 1.) / 5.

let rec biter f acc i =
    if abs (f i) > epsilon then
        acc + f i + biter f acc (i + 1.0)
    else
        acc

// Dumb Tailor (not using previous equations)
let rec dumb_tailor_eq x acc n =
    let crnt = funct_n x n

    if abs (crnt) > epsilon then
        dumb_tailor_eq x (acc + crnt) (n + 1)
    else
        (acc, n)


let dumb_tailor x = dumb_tailor_eq x 0. 1


//Smart tailor, using previous equations

let rec smart_tailor_eq x summ n prev =
    // формула зависит не от члена, а от пред. суммы
    let crnt = funct_prev x summ

    if abs (summ - crnt) > epsilon then
        smart_tailor_eq x (crnt) (n + 1) summ
    else
        (summ, n)

let smart_tailor x = smart_tailor_eq x 0 0 0

let programm steps =
    printfn "|===============================================================================|"
    printfn "|     X        Builtin        Dumb Taylor     # terms     Smart Taylor   # terms|"
    printfn "|===============================================================================|"

    for i = 0 to steps do
        let x = (2. / float steps) * float i
        printf "|  %3.5f" (x)
        printf "\t%3.5f" (funct x)
        let first_f = dumb_tailor x

        match first_f with
        | (floatValue, intValue) ->
            printf "\t%3.5f" floatValue
            printf "\t%4d " (intValue - 1) // нужно начинать с 1 для корректности

        let clever = smart_tailor x

        match clever with
        | (fs, intVal) ->
            printf "\t  %3.5f" fs
            printf "\t%4d " intVal

        printfn "\t|"

programm 10
