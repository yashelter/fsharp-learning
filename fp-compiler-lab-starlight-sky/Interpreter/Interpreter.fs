namespace Interpreter
open System.IO

module Interpreter =
    let mutable tracing = false
    type Name = string

    type Construction =
        | Float of float
        | Bool of bool
        | String of string
        | DefineVarible of Construction * Construction // detect
        | DefineFunction of Name * List<Construction> * Construction
        | Func of List<Construction> * Construction
        | Function of Name * List<Construction>
        | Varible of Name
        | OtherContext of Construction * Construction
        | BasicOperation of Name * left_construction:Construction * right_construction:Construction
        | LibraryOperation of Name * Construction
        | CatalogOperaionConcat of  Construction * Construction
        | CatalogOperaionAppend of  Construction * Construction
        | ReadFromFile of Name // DefineVarible("x", ReadFromFile(path))
        | WriteToFile of Name * Construction
        | Condition of Construction * Construction * Construction
        | Eval of List<Construction>
        | ListExpr of List<Construction>
        | Catalog of List<Construction>
        | CatalogOperaion of Name * Construction
        | Keyword of string
        | Nothing
    and
        Context = Map<Name, Construction>

    let solve_basic (eq : Construction) : Construction =
    
        match eq with
        | Bool(x) -> Bool(x)
        | BasicOperation("+", Float(x), Float(y)) -> Float(x+y)
        | BasicOperation("-", Float(x), Float(y)) -> Float(x-y)
        | BasicOperation("*", Float(x), Float(y)) -> Float(x*y)
        | BasicOperation("/", Float(x), Float(y)) -> Float(x/y)
        | BasicOperation("!=", Float(x), Float(y)) -> 
            if x<>y then Bool(true)
            else Bool(false)
        | BasicOperation("==", Float(x), Float(y)) -> 
            if x=y then Bool(true)
            else Bool(false)
        | BasicOperation("!=", String(x), String(y)) -> 
            if x<>y then Bool(true)
            else Bool(false)
        | BasicOperation("==", String(x), String(y)) -> 
            if x=y then Bool(true)
            else Bool(false)
        | BasicOperation("!=", Bool(x), Bool(y)) -> 
            if x<>y then Bool(true)
            else Bool(false)
        | BasicOperation("==", Bool(x), Bool(y)) -> 
            if x=y then Bool(true)
            else Bool(false)
        | BasicOperation("!=", Catalog(x), Catalog(y)) -> 
            if x<>y then Bool(true)
            else Bool(false)
        | BasicOperation("==", Catalog([]), Catalog([])) -> 
            Bool(true)
        | BasicOperation("==", Catalog(x), Catalog(y)) -> 
            if x=y then Bool(true)
            else Bool(false)
        | BasicOperation(">", Float(x), Float(y)) -> 
            if x>y then Bool(true)
            else Bool(false)
        | BasicOperation("<", Float(x), Float(y)) -> 
            if x<y then Bool(true)
            else Bool(false)
        | BasicOperation("<=", Float(x), Float(y)) -> 
            if x<=y then Bool(true)
            else Bool(false)
        | BasicOperation(">=", Float(x), Float(y)) -> 
            if x<=y then Bool(true)
            else Bool(false)
        | _ as ex -> 
            if tracing = true then printfn "[error] Try to use %A" ex
            failwith("Non implemented operation")

    let solve_library (eq : Construction) : Construction =
        match eq with
        | LibraryOperation("print", Float(value)) -> 
            printf "%f" value
            Nothing

        | LibraryOperation("printfn", Float(value)) -> 
            printfn "%f" value
            Nothing

        | LibraryOperation("print", String(value)) -> 
            printf "%s" value
            Nothing

        | LibraryOperation("printfn", String(value)) -> 
            printfn "%s" value
            Nothing

        | LibraryOperation("print", arg)-> 
            if tracing=true then printfn ("[warning] Not implemented print for this type" )
            printfn "%A" arg
            Nothing

        | LibraryOperation("printfn", arg)-> 
            if tracing=true then printfn "[warning] Not implemented print for this type %A" arg
            printfn "%A" arg
            Nothing

        | _ as x-> 
            if tracing=true then printfn "[error] Not exsist function %A" x
            Nothing

    let solve_list (eq : Construction) : Construction =
        match eq with 
        | CatalogOperaion("head", Catalog(lst)) -> List.head lst
        | CatalogOperaion("tail", Catalog(lst)) -> (Catalog(List.tail lst))
        | _ as ex->
            if tracing=true then printfn "[error] Try to call list function %A" ex 
            failwith ("Not implemented operation")

    let writeToFile (filePath: string) (content: string) =
        try
            use writer = new StreamWriter(filePath)
            writer.Write(content)
        with
        | ex ->
            if tracing=true then
                printfn "[error] Getted error while writed: %s" ex.Message
            failwith "Error while writing into file"

    let readFromFile (filePath: string) : Construction =
        try
            use reader = new StreamReader(filePath)
            String(reader.ReadToEnd())
        with
        | ex ->
            if tracing = true then
                printfn "[error] Getted error while readed: %s" ex.Message
            failwith "Getted error while readed file"
            Nothing

    let rec eval (program:Construction) (context: Context) : Construction =
        if tracing=true then 
            printfn "[trace]"
            printfn "%A" program
            printfn ""

        match program with
        | CatalogOperaion(name, list) -> 
            let list' = eval list context
            solve_list (CatalogOperaion (name, list'))

        | LibraryOperation(name, argument) ->
            let arg' = eval argument context
            solve_library (LibraryOperation(name, arg'))

        | ReadFromFile(path) ->
            let res = readFromFile path
            res
        
        | WriteToFile(path, whatWrite) ->
            writeToFile path (string whatWrite)
            Nothing
         
        | CatalogOperaionConcat(lst1, lst2) ->
            let lst1', lst2' = eval lst1 context, eval lst2 context
            match lst1' with
                | Catalog(x) ->
                    match lst2' with
                        | Catalog(y) ->
                            let x' = eval_list x context
                            let y' = eval_list y context
                            Catalog(x'@y')
                            
                        | _ -> failwith ("unexpexted argument")
                 | _ -> 
                    if tracing=true 
                        then printfn "[error] wrong concat with %A" (CatalogOperaionConcat(lst1, lst2))
                    failwith ("unexpexted argument")
        
        | CatalogOperaionAppend(elem, lst) ->
            let elem', lst' = eval elem context, eval lst context
            match lst' with
                | Catalog(x) ->
                    let evaled_list = eval_list x context
                    let new_ev_list = List.insertAt 0 elem' evaled_list
                    Catalog(new_ev_list)
                 | _ -> 
                    if tracing=true 
                        then printfn "[error] wrong append with %A" (CatalogOperaionAppend(elem, lst))
                    failwith ("unexpexted argument")
        
        | Float(x) -> Float(x)

        | Bool(x) -> Bool(x)

        | String(x) -> String(x)

        | Catalog(lst) -> Catalog(lst)

        | BasicOperation(name, left, right) -> 
            let left' = eval left context
            let right' = eval right context
            solve_basic(BasicOperation(name, left', right'))

        | Varible(x) -> 
            match Map.tryFind x context with
                | Some(value) -> eval value context
                | None -> 
                    if tracing=true then printfn "[error] Varible name was %A" x
                    failwith("Not exist varible with given name")

        | OtherContext(left, right) -> 
            let (context', result_l) = get_new_context left context
            let result_r = eval right context'
            if result_r = Nothing then result_l
            else result_l

        | Condition(condition, iftrue, ifelse) -> 
            let brach_choose = eval condition context
            if brach_choose=Bool(true) then  
                eval iftrue context
            elif brach_choose=Bool(false) then eval ifelse context
            else 
                if tracing=true then 
                    printfn "[error] Condition returned %A" brach_choose
                failwith("Conditon not returned bool type")

        | Function(name, vars_values) -> 
            match Map.tryFind name context with
                | Some(Func(vars_names, to_do)) -> 
                    if List.length vars_names <> List.length vars_values then
                        if tracing=true then printfn "Problem with args with funciton %A" name
                        failwith("wrong function parameters")
                    else
                        let new_context = add_varibles vars_names vars_values context
                        eval to_do new_context

                | None -> 
                    if tracing=true then 
                        printfn "Given not existing function name was %A" name
                    failwith("Not exist function with given name")
                | _ as tp -> 
                    if tracing=true then printfn "[error] Getted type was %A" tp
                    failwith("Unexpected type")
        | Nothing -> Nothing
        | Func(_) as t' -> t'
        | _ as x-> 
            if tracing=true then printfn "[error] Wrong parsing format with %A" x 
            failwith ("Wrong parsing format")

    and get_new_context constuction context : (Context * Construction) =
        match constuction with
        | DefineVarible(Varible(name), v_type) -> (Map.add name v_type context, Nothing)
        | DefineFunction(name, lst, to_do) ->  (Map.add name (Func(lst, to_do)) context, Nothing)
        | _ ->
            (context, eval constuction context)
            

    and add_varibles (names:List<Construction>) (vals:List<Construction>) (context:Context) =
        if names = [] then context
        else 
            let head = List.head names
            match head with
                | Varible(id) -> add_varibles (List.tail names) (List.tail vals) (Map.add id (eval (List.head vals) context) context)
                | _ as arg-> 
                    if tracing=true then printfn "[error] Argument was %A" arg
                    failwith ("unexpexted function argument")
    and eval_list (lst : List<Construction> ) (context : Context) : List<Construction> =
        match lst with
            | [] -> []
            | h::t -> (eval h context)::(eval_list t context)


// DefineFunc ("fact", [Var "x"], BasicOperation ("==", Var "x,", Float 1.0)) (1)
// Function ("factorial", [Float 10.0]) (2)
// Function ("factorial", [Float 101.0]) (3)
(*

eval (OtherContext(DefineFunction("pow2", ["x"], BasicOperation("*", Varible("x"), Varible("x"))), Function("pow2", [Float(4)]))) Map.empty

eval (OtherContext(DefineVarible(Varible("x"), Float(0)),
    Condition((BasicOperation("!=", Varible("x"), Float(0)),(Float(5)), 
    (BasicOperation("*", Float(5),Float(5))))))) Map.empty


eval (OtherContext(DefineFunction("factorial", 
    [Varible("x")], 
    Condition(
        (BasicOperation("==", Varible("x"), Float(1))), 
            (Float(1)), 
            (BasicOperation("*", Varible("x"), 
                Function("factorial", [BasicOperation("-", Varible("x"), Float(1))])
                )
             )
        )), OtherContext(Function("factorial", [Float(7)]),Function("factorial", [Float(6)])))) Map.empty

        *)