namespace Parser

open FParsec
open System.IO
open Interpreter.Interpreter
open Parser.Parser

module ParserCore =
    type id = string

    let tracingparseFileConstruct (filePath: string) =
        let text = File.ReadAllText(filePath)
        match Parser.ParseString text with
        | Result.Ok parse -> 
            match parse with
            | Eval lst ->
                let otherContext = transformToOtherContext parse 
                printfn "%A" otherContext
            | _ -> printfn "Parsed expression is not a list expression"
        | Result.Error err ->
            printfn "%s" err
    
    let rec readAllFiles (list : string list) (acc : string) =
        match list with
            | [] -> acc
            | path::otherl -> 
                 let text = File.ReadAllText(path)
                 let newText =  text + "\n" + acc 
                 readAllFiles otherl newText

    let parseFileConstruct (text: string) : Construction option =
            match Parser.ParseString text with
            | Result.Ok parse -> 
                match parse with
                | Eval lst ->
                    let otherContext = transformToOtherContext parse 
                    Some otherContext
                | _ -> None 
            | Result.Error err ->
                printfn "%s" err
                None 

