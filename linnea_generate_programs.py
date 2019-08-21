if __name__ == "__main__":

    import random
    import os
    import shutil
    import fnmatch
    import linnea.config

    #This function used to do something, but right now it's not needed anymore. Remove?
    linnea.config.init()
    # linnea.config.set_verbosity(2)

    # from linnea.derivation.graph.constructive import DerivationGraph
    # from linnea.derivation.graph.exhaustive import DerivationGraph
    from linnea.derivation.graph.derivation import DerivationGraph


    import linnea.examples.examples
    import linnea.examples.application

    RESTART = True
    if(RESTART):
        if os.path.exists("Programs/"):
            shutil.rmtree("Programs/")
        os.mkdir("Programs/")

    expressions = ["01", "02", "03", "04", "05",
                   "06", "07", "08", "09", "10",
                   "11", "12", "13", "14", "15",
                   "16", "17", "18", "19", "20",
                   "21", "22", "23", "24", "25"]

    total_algo = 0
    for e in expressions:
        print("Evaluating expression" + e)

        example = eval("linnea.examples.application.Example" + e + "()")


        #example = linnea.examples.application.Example05()
        #example = linnea.examples.application.Example03()

        # example = linnea.examples.examples.Example113()
        # example = linnea.examples.examples.Example047()
        # example = linnea.examples.examples.Example135()
        # example = linnea.examples.examples.Example063()
        # example = linnea.examples.examples.Example007()
        # example = linnea.examples.examples.Example154() # linear system
        # example = linnea.examples.examples.Example158() # complex inverse
        # example = linnea.examples.examples.Example036() # complex inverse

        # example = linnea.examples.examples.Example163() # redundancy, variants
        # example = linnea.examples.examples.Example054()

        print(example.eqns)
        #print(example.eqns.__dict__)

        # from linnea.derivation.graph.utils import generate_variants
        # for eqns in generate_variants(example.eqns.to_normalform(), 1):
        #     print(eqns)
        # quit()

        # from linnea.code_generation.experiments import operand_generation, runner, reference_code
        # operand_generation.generate_operand_generator("tmp", example.eqns)
        # reference_code.generate_reference_code("tmp", example.eqns)
        # quit()


        graph = DerivationGraph(example.eqns)
        #print(graph.nodes[0].__dict__)
        # trace = graph.derivation(
        #                     solution_nodes_limit=100,
        #                     iteration_limit=20,
        #                     merging=True,
        #                     dead_ends=True)
        # print(":".join(str(t) for t in trace))

        trace = graph.derivation(time_limit=5)
        print(trace)

        # import linnea.temporaries
        # print("\n".join(["{}: {}".format(k, v) for k, v in linnea.temporaries._table_of_temporaries.items()]))
        # print("\n".join(["{}: {}".format(k, v) for k, v in linnea.temporaries._equivalent_expressions.items()]))

        graph.write_output(code=True,
                           derivation=True,
                           output_name="tmp",
                           experiment_code=True,
                           algorithms_limit=100,
                           graph=True,
                           graph_style=linnea.config.GraphStyle.full)

        print("Generated algorithms for Example" + e)

        dest = "Programs/Example"+e+"/"
        shutil.copytree("tmp/", dest)

        print("Copied to " + dest)

        num_algo = len(fnmatch.filter(os.listdir(dest + "Julia/generated/"), '*.jl'))
        print("Generated ALgorithms : ", num_algo)
        print("...................")
        total_algo += num_algo

        #exit(code=-1)

    print("Total Algorithms Generated : ", total_algo)
