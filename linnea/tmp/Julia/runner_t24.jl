using Test
using Logging
using MatrixGenerator

using LinearAlgebra.BLAS
BLAS.set_num_threads(24)

include("operand_generator.jl")
include("experiments/algorithm0.jl")
include("reference/naive.jl")
include("reference/recommended.jl")

function main()
    matrices = operand_generator()

    @info("Performing Test run...")
    result_naive = collect(naive(map(copy, matrices)...)[1])
    result_recommended = collect(recommended(map(copy, matrices)...)[1])
    result_algorithm0 = collect(algorithm0(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    @test isapprox(result_recommended, result_naive, rtol=1e-3)
    @test isapprox(result_algorithm0, result_recommended, rtol=1e-3)
    @info("Test run performed successfully")


    @info("Running Benchmarks...")
    plotter = Benchmarker.Plot("julia_results_tmp", ["algorithm"; "threads"]);
    Benchmarker.add_data(plotter, ["algorithm0"; 24], Benchmarker.measure(20, algorithm0, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["naive_julia"; 24], Benchmarker.measure(20, naive, matrices...) );
    Benchmarker.add_data(plotter, ["recommended_julia"; 24], Benchmarker.measure(20, recommended, matrices...) );
    Benchmarker.finish(plotter);
    @info("Benchmarks complete")
end

main()
