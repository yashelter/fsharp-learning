open Interpreter.Interpreter
open Parser.ParserCore


[<EntryPoint>]
let main args :int=
    if args.Length > 2 && args[0]="tracing=true" then
        tracing<-true

    if args.Length < 1 then
        printf "Incorrect usage of program"
        1
    else
        try
            let mutable paths = Array.toList args
            if (tracing=true) then
                paths <- Array.toList (Array.tail args)

            let finalfile = readAllFiles paths ""
            let result = parseFileConstruct finalfile
            let eval'ed = eval (result.Value) Map.empty
            
            if tracing=true then printfn "%A" eval'ed
            0
        with
        | :? System.NullReferenceException as ex ->
            printfn "Parsing was failed"
            2
        | ex ->
            printfn "Occured unprocessed exception: %A" ex
            2
