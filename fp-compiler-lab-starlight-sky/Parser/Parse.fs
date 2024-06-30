namespace Parser
open FParsec
open System.IO
open Interpreter.Interpreter


module Parser =
    
    let rec processList (lst: Construction list) =
        match lst with
        | [] -> Nothing
        | x::y::tail -> OtherContext(x, processList (y::tail))
        | [x] -> OtherContext (x, Nothing)
        | _ -> failwith ("List must contain at least two elements")

    let transformToOtherContext (node: Construction) =
        match node with
        | Eval lst -> processList lst
        | _ -> failwith ("Node is not a list expression")
        
   
    let DifferParse, OperRef = createParserForwardedToRef()
    
    let private Comments = pchar '*' >>. manyChars (noneOf "*") >>. pchar '*' >>. spaces
   
    let private meSpaces = Comments <|> spaces

    let private _List = pstring "catalog" >>. meSpaces .>> skipChar '(' >>. meSpaces >>. 
                         (attempt (sepBy DifferParse (pstring "|" .>> meSpaces)) <|> sepEndBy1 DifferParse (pstring "|" .>> meSpaces))
                            .>> meSpaces .>> skipChar ')' .>> meSpaces |>> Catalog
    let private _BoolTrue = pstring "true" >>. meSpaces |>> (fun _ -> Bool true)
    let private _BoolFalse = pstring "false" >>. meSpaces |>> (fun _ -> Bool false)
    let private _Bool = _BoolFalse <|> _BoolTrue // объединить два парсера в один (альтернатива)

    let private Number = pfloat .>> meSpaces |>> Float 
    let private Variable =
        many1Satisfy2 (fun c -> System.Char.IsLetter(c) || c = '_') (fun c -> System.Char.IsLetterOrDigit(c) || c = '_')
        |>> Varible .>> meSpaces

    let private manyCharsBetween popen pclose pchar = popen >>? manyCharsTill pchar pclose
    let private anyStringBetween popen pclose = manyCharsBetween popen pclose anyChar
    let private quotedString = skipChar '"' |> anyStringBetween <| skipChar '"'

    let private _String = quotedString .>> meSpaces |>> String
    let private If_keyword = stringReturn "if" <| Keyword "if" .>> meSpaces
    let private Else_keyword = stringReturn "else" <| Keyword "else" .>> meSpaces
    let private Then_keyword = stringReturn "=>" <| Keyword "then" .>> meSpaces

    let private NumberList = pstring "[" >>. meSpaces >>. sepBy (DifferParse) (pstring "," .>> meSpaces) .>> pstring "]" .>> meSpaces 
    let private FunctionDoll = skipChar '$' .>> meSpaces >>. many1Chars (noneOf " []\n") .>> meSpaces .>>. NumberList |>> Function
    let private Val_keyword  = pstring "detect" >>. meSpaces >>. Variable .>> meSpaces .>> skipChar '=' .>> meSpaces .>>. DifferParse .>> meSpaces |>> DefineVarible
    let private LibraryOpP  = pstring "print" .>> meSpaces .>>. DifferParse .>> meSpaces |>> LibraryOperation
    let private LibraryOpPFn  = pstring "printfn" .>> meSpaces .>>. DifferParse .>> meSpaces |>> LibraryOperation
    let private Print = LibraryOpPFn <|> LibraryOpP

    // Parcing operations

    let private Equal = pstring "eq" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("==", left, right))
    let private NotEqual = pstring "neq" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("!=", left, right))
    let private Mult = pstring "mul" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("*", left, right))
    let private Divide = pstring "div" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("/", left, right))
    let private Add = pstring "add" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("+", left, right))
    let private Subtr = pstring "sub" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("-", left, right))
    let private GreaterThan = pstring "gt" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation (">", left, right))
    let private LessThan = pstring "lt" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("<", left, right))
    let private GreaterThanE = pstring "gte" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation (">=", left, right))
    let private LessThanE = pstring "lte" >>. meSpaces >>. skipChar '(' >>. meSpaces >>. DifferParse .>> skipChar ',' .>> meSpaces .>>. DifferParse .>> skipChar ')' .>> meSpaces |>> (fun (left, right) -> BasicOperation ("<=", left, right))
    
    
    let private CatalogOPfirst = (_List <|> Variable) .>> meSpaces .>>. (pstring "cfirst") .>> meSpaces |>> (fun (a, _) -> CatalogOperaion ("head", a))
    let private CatalogOPlast = (_List <|> Variable) .>> meSpaces .>>. (pstring "ctail") .>> meSpaces |>> (fun (a, _) -> CatalogOperaion ("tail", a))
    let private CatalogOPConcat = _List .>> meSpaces .>> pstring "<->" .>> meSpaces .>>. _List .>> meSpaces |>> CatalogOperaionConcat
    let private CatalogOP =  attempt CatalogOPfirst <|> CatalogOPlast
    let private CatalogOPAppend = (CatalogOP <|> FunctionDoll <|> attempt _List <|> attempt Variable <|> attempt Number <|> attempt _Bool <|> attempt _String) .>> meSpaces .>> pstring "->>" .>> meSpaces .>>. DifferParse .>> meSpaces |>> CatalogOperaionAppend

    
    
    
    let private FileOpen = pstring "ReadFromFile" >>. meSpaces >>. quotedString |>> ReadFromFile 
    let private FileOpenWrite = pstring "WriteToFile" >>. meSpaces >>. quotedString .>> meSpaces .>>. DifferParse |>> WriteToFile 

    let private Keyword = If_keyword <|> Else_keyword <|> Then_keyword
    let private Oper = Equal <|> Mult <|> Divide <|> Add <|> Subtr <|> NotEqual <|> attempt LessThanE <|> attempt GreaterThanE <|> attempt LessThan <|> attempt GreaterThan
    let private Conditions = Keyword >>. DifferParse .>> Keyword .>>. DifferParse .>> Keyword .>>. DifferParse |>> (fun ((a, b), c) -> Condition (a, b, c))
    let private Lambda_keyword = pstring "lambda" >>. meSpaces >>. many1Chars (noneOf "\"\\ []\n") .>> meSpaces .>>. NumberList .>> skipChar '=' .>> meSpaces .>> skipChar '{' .>> meSpaces .>>. (sepEndBy1 DifferParse spaces) .>> meSpaces .>> skipChar '}' .>> meSpaces |>> (fun ((a, b), c) -> DefineFunction (a, b, transformToOtherContext ((Eval) c)))



    OperRef.Value <- 
        choice [
            attempt _Bool
            attempt Oper
            attempt Lambda_keyword
            attempt Val_keyword
            attempt Conditions
            attempt Print
            attempt CatalogOPAppend
            attempt CatalogOP
            attempt CatalogOPConcat
            attempt _List
            attempt FunctionDoll
            attempt FileOpenWrite
            attempt FileOpen
            attempt Number
            attempt Variable
            attempt Keyword
            attempt _String
        ]

    let private result_pars = spaces >>. meSpaces >>. many DifferParse .>> eof |>> Eval

    let ParseString (s: string): Result<Construction, string> =
        match run result_pars s with
        | Success (res, _, _) -> Result.Ok res
        | Failure (err, _, _) ->Result.Error err