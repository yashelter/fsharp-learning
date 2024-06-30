open Interpreter.Interpreter

let case1 = (OtherContext
    (DefineVarible (Varible "a", Catalog [Float 1.0]),
     OtherContext
       (DefineVarible (Varible "b", Catalog []),
        OtherContext
          (DefineFunction
             ("map", [Varible "f"; Varible "c"],
              OtherContext
                (Condition
                   (BasicOperation ("==", Varible "c", Catalog []), Catalog [],
                    CatalogOperaionAppend
                      (Function ("f", [CatalogOperaion ("head", Varible "c")]),
                       Function
                            ("map",
                             [Varible "f"; CatalogOperaion ("tail", Varible "c")]))),
                 Nothing)),
           OtherContext
             (DefineFunction
                ("t", [Varible "x"],
                 OtherContext
                   (BasicOperation ("*", Varible "x", Varible "x"), Nothing)),
              OtherContext
                (LibraryOperation
                   ("printfn",
                    Function
                      ("map",
                       [Varible "t"; Catalog [Float 1.0; Float 2.0; Float 10.0]])),
                 Nothing))))))

let test1 = eval case1 Map.empty