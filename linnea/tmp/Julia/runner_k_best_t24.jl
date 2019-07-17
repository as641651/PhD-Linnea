using Test
using Logging
using MatrixGenerator

using LinearAlgebra.BLAS
BLAS.set_num_threads(24)

include("operand_generator.jl")
include("experiments/algorithm0.jl")
include("experiments/algorithm1.jl")
include("experiments/algorithm2.jl")
include("experiments/algorithm3.jl")
include("experiments/algorithm4.jl")
include("experiments/algorithm5.jl")
include("experiments/algorithm6.jl")
include("experiments/algorithm7.jl")
include("experiments/algorithm8.jl")
include("experiments/algorithm9.jl")
include("experiments/algorithm10.jl")
include("experiments/algorithm11.jl")
include("experiments/algorithm12.jl")
include("experiments/algorithm13.jl")
include("experiments/algorithm14.jl")
include("experiments/algorithm15.jl")
include("experiments/algorithm16.jl")
include("experiments/algorithm17.jl")
include("experiments/algorithm18.jl")
include("experiments/algorithm19.jl")
include("experiments/algorithm20.jl")
include("experiments/algorithm21.jl")
include("experiments/algorithm22.jl")
include("experiments/algorithm23.jl")
include("experiments/algorithm24.jl")
include("experiments/algorithm25.jl")
include("experiments/algorithm26.jl")
include("experiments/algorithm27.jl")
include("experiments/algorithm28.jl")
include("experiments/algorithm29.jl")
include("experiments/algorithm30.jl")
include("experiments/algorithm31.jl")
include("experiments/algorithm32.jl")
include("experiments/algorithm33.jl")
include("experiments/algorithm34.jl")
include("experiments/algorithm35.jl")
include("experiments/algorithm36.jl")
include("experiments/algorithm37.jl")
include("experiments/algorithm38.jl")
include("experiments/algorithm39.jl")
include("experiments/algorithm40.jl")
include("experiments/algorithm41.jl")
include("experiments/algorithm42.jl")
include("experiments/algorithm43.jl")
include("experiments/algorithm44.jl")
include("experiments/algorithm45.jl")
include("experiments/algorithm46.jl")
include("experiments/algorithm47.jl")
include("experiments/algorithm48.jl")
include("experiments/algorithm49.jl")
include("experiments/algorithm50.jl")
include("experiments/algorithm51.jl")
include("experiments/algorithm52.jl")
include("experiments/algorithm53.jl")
include("experiments/algorithm54.jl")
include("experiments/algorithm55.jl")
include("experiments/algorithm56.jl")
include("experiments/algorithm57.jl")
include("experiments/algorithm58.jl")
include("experiments/algorithm59.jl")
include("experiments/algorithm60.jl")
include("experiments/algorithm61.jl")
include("experiments/algorithm62.jl")
include("experiments/algorithm63.jl")
include("experiments/algorithm64.jl")
include("experiments/algorithm65.jl")
include("experiments/algorithm66.jl")
include("experiments/algorithm67.jl")
include("experiments/algorithm68.jl")
include("experiments/algorithm69.jl")
include("experiments/algorithm70.jl")
include("experiments/algorithm71.jl")
include("experiments/algorithm72.jl")
include("experiments/algorithm73.jl")
include("experiments/algorithm74.jl")
include("experiments/algorithm75.jl")
include("experiments/algorithm76.jl")
include("experiments/algorithm77.jl")
include("experiments/algorithm78.jl")
include("experiments/algorithm79.jl")
include("experiments/algorithm80.jl")
include("experiments/algorithm81.jl")
include("experiments/algorithm82.jl")
include("experiments/algorithm83.jl")
include("experiments/algorithm84.jl")
include("experiments/algorithm85.jl")
include("experiments/algorithm86.jl")
include("experiments/algorithm87.jl")
include("experiments/algorithm88.jl")
include("experiments/algorithm89.jl")
include("experiments/algorithm90.jl")
include("experiments/algorithm91.jl")
include("experiments/algorithm92.jl")
include("experiments/algorithm93.jl")
include("experiments/algorithm94.jl")
include("experiments/algorithm95.jl")
include("experiments/algorithm96.jl")
include("experiments/algorithm97.jl")
include("experiments/algorithm98.jl")
include("experiments/algorithm99.jl")
include("reference/naive.jl")
include("reference/recommended.jl")

function main()
    matrices = operand_generator()

    @info("Performing Test run...")
    result_naive = collect(naive(map(copy, matrices)...)[1])
    result_recommended = collect(recommended(map(copy, matrices)...)[1])
    result_algorithm0 = collect(algorithm0(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm1 = collect(algorithm1(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm2 = collect(algorithm2(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm3 = collect(algorithm3(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm4 = collect(algorithm4(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm5 = collect(algorithm5(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm6 = collect(algorithm6(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm7 = collect(algorithm7(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm8 = collect(algorithm8(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm9 = collect(algorithm9(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm10 = collect(algorithm10(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm11 = collect(algorithm11(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm12 = collect(algorithm12(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm13 = collect(algorithm13(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm14 = collect(algorithm14(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm15 = collect(algorithm15(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm16 = collect(algorithm16(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm17 = collect(algorithm17(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm18 = collect(algorithm18(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm19 = collect(algorithm19(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm20 = collect(algorithm20(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm21 = collect(algorithm21(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm22 = collect(algorithm22(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm23 = collect(algorithm23(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm24 = collect(algorithm24(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm25 = collect(algorithm25(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm26 = collect(algorithm26(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm27 = collect(algorithm27(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm28 = collect(algorithm28(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm29 = collect(algorithm29(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm30 = collect(algorithm30(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm31 = collect(algorithm31(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm32 = collect(algorithm32(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm33 = collect(algorithm33(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm34 = collect(algorithm34(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm35 = collect(algorithm35(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm36 = collect(algorithm36(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm37 = collect(algorithm37(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm38 = collect(algorithm38(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm39 = collect(algorithm39(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm40 = collect(algorithm40(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm41 = collect(algorithm41(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm42 = collect(algorithm42(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm43 = collect(algorithm43(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm44 = collect(algorithm44(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm45 = collect(algorithm45(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm46 = collect(algorithm46(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm47 = collect(algorithm47(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm48 = collect(algorithm48(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm49 = collect(algorithm49(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm50 = collect(algorithm50(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm51 = collect(algorithm51(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm52 = collect(algorithm52(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm53 = collect(algorithm53(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm54 = collect(algorithm54(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm55 = collect(algorithm55(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm56 = collect(algorithm56(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm57 = collect(algorithm57(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm58 = collect(algorithm58(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm59 = collect(algorithm59(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm60 = collect(algorithm60(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm61 = collect(algorithm61(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm62 = collect(algorithm62(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm63 = collect(algorithm63(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm64 = collect(algorithm64(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm65 = collect(algorithm65(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm66 = collect(algorithm66(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm67 = collect(algorithm67(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm68 = collect(algorithm68(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm69 = collect(algorithm69(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm70 = collect(algorithm70(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm71 = collect(algorithm71(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm72 = collect(algorithm72(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm73 = collect(algorithm73(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm74 = collect(algorithm74(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm75 = collect(algorithm75(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm76 = collect(algorithm76(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm77 = collect(algorithm77(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm78 = collect(algorithm78(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm79 = collect(algorithm79(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm80 = collect(algorithm80(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm81 = collect(algorithm81(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm82 = collect(algorithm82(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm83 = collect(algorithm83(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm84 = collect(algorithm84(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm85 = collect(algorithm85(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm86 = collect(algorithm86(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm87 = collect(algorithm87(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm88 = collect(algorithm88(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm89 = collect(algorithm89(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm90 = collect(algorithm90(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm91 = collect(algorithm91(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm92 = collect(algorithm92(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm93 = collect(algorithm93(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm94 = collect(algorithm94(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm95 = collect(algorithm95(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm96 = collect(algorithm96(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm97 = collect(algorithm97(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm98 = collect(algorithm98(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    result_algorithm99 = collect(algorithm99(map(MatrixGenerator.unwrap, map(copy, matrices))...)[1])
    @test isapprox(result_recommended, result_naive, rtol=1e-3)
    @test isapprox(result_algorithm0, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm1, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm2, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm3, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm4, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm5, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm6, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm7, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm8, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm9, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm10, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm11, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm12, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm13, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm14, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm15, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm16, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm17, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm18, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm19, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm20, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm21, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm22, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm23, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm24, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm25, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm26, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm27, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm28, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm29, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm30, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm31, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm32, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm33, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm34, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm35, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm36, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm37, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm38, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm39, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm40, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm41, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm42, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm43, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm44, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm45, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm46, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm47, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm48, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm49, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm50, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm51, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm52, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm53, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm54, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm55, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm56, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm57, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm58, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm59, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm60, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm61, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm62, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm63, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm64, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm65, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm66, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm67, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm68, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm69, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm70, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm71, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm72, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm73, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm74, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm75, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm76, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm77, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm78, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm79, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm80, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm81, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm82, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm83, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm84, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm85, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm86, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm87, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm88, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm89, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm90, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm91, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm92, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm93, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm94, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm95, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm96, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm97, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm98, result_recommended, rtol=1e-3)
    @test isapprox(result_algorithm99, result_recommended, rtol=1e-3)
    @info("Test run performed successfully")


    @info("Running Benchmarks...")
    plotter = Benchmarker.Plot("julia_results_tmp", ["algorithm"; "threads"]);
    Benchmarker.add_data(plotter, ["algorithm0"; 24], Benchmarker.measure(20, algorithm0, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm1"; 24], Benchmarker.measure(20, algorithm1, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm2"; 24], Benchmarker.measure(20, algorithm2, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm3"; 24], Benchmarker.measure(20, algorithm3, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm4"; 24], Benchmarker.measure(20, algorithm4, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm5"; 24], Benchmarker.measure(20, algorithm5, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm6"; 24], Benchmarker.measure(20, algorithm6, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm7"; 24], Benchmarker.measure(20, algorithm7, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm8"; 24], Benchmarker.measure(20, algorithm8, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm9"; 24], Benchmarker.measure(20, algorithm9, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm10"; 24], Benchmarker.measure(20, algorithm10, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm11"; 24], Benchmarker.measure(20, algorithm11, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm12"; 24], Benchmarker.measure(20, algorithm12, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm13"; 24], Benchmarker.measure(20, algorithm13, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm14"; 24], Benchmarker.measure(20, algorithm14, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm15"; 24], Benchmarker.measure(20, algorithm15, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm16"; 24], Benchmarker.measure(20, algorithm16, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm17"; 24], Benchmarker.measure(20, algorithm17, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm18"; 24], Benchmarker.measure(20, algorithm18, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm19"; 24], Benchmarker.measure(20, algorithm19, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm20"; 24], Benchmarker.measure(20, algorithm20, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm21"; 24], Benchmarker.measure(20, algorithm21, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm22"; 24], Benchmarker.measure(20, algorithm22, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm23"; 24], Benchmarker.measure(20, algorithm23, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm24"; 24], Benchmarker.measure(20, algorithm24, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm25"; 24], Benchmarker.measure(20, algorithm25, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm26"; 24], Benchmarker.measure(20, algorithm26, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm27"; 24], Benchmarker.measure(20, algorithm27, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm28"; 24], Benchmarker.measure(20, algorithm28, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm29"; 24], Benchmarker.measure(20, algorithm29, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm30"; 24], Benchmarker.measure(20, algorithm30, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm31"; 24], Benchmarker.measure(20, algorithm31, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm32"; 24], Benchmarker.measure(20, algorithm32, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm33"; 24], Benchmarker.measure(20, algorithm33, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm34"; 24], Benchmarker.measure(20, algorithm34, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm35"; 24], Benchmarker.measure(20, algorithm35, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm36"; 24], Benchmarker.measure(20, algorithm36, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm37"; 24], Benchmarker.measure(20, algorithm37, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm38"; 24], Benchmarker.measure(20, algorithm38, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm39"; 24], Benchmarker.measure(20, algorithm39, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm40"; 24], Benchmarker.measure(20, algorithm40, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm41"; 24], Benchmarker.measure(20, algorithm41, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm42"; 24], Benchmarker.measure(20, algorithm42, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm43"; 24], Benchmarker.measure(20, algorithm43, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm44"; 24], Benchmarker.measure(20, algorithm44, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm45"; 24], Benchmarker.measure(20, algorithm45, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm46"; 24], Benchmarker.measure(20, algorithm46, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm47"; 24], Benchmarker.measure(20, algorithm47, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm48"; 24], Benchmarker.measure(20, algorithm48, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm49"; 24], Benchmarker.measure(20, algorithm49, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm50"; 24], Benchmarker.measure(20, algorithm50, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm51"; 24], Benchmarker.measure(20, algorithm51, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm52"; 24], Benchmarker.measure(20, algorithm52, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm53"; 24], Benchmarker.measure(20, algorithm53, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm54"; 24], Benchmarker.measure(20, algorithm54, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm55"; 24], Benchmarker.measure(20, algorithm55, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm56"; 24], Benchmarker.measure(20, algorithm56, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm57"; 24], Benchmarker.measure(20, algorithm57, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm58"; 24], Benchmarker.measure(20, algorithm58, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm59"; 24], Benchmarker.measure(20, algorithm59, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm60"; 24], Benchmarker.measure(20, algorithm60, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm61"; 24], Benchmarker.measure(20, algorithm61, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm62"; 24], Benchmarker.measure(20, algorithm62, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm63"; 24], Benchmarker.measure(20, algorithm63, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm64"; 24], Benchmarker.measure(20, algorithm64, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm65"; 24], Benchmarker.measure(20, algorithm65, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm66"; 24], Benchmarker.measure(20, algorithm66, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm67"; 24], Benchmarker.measure(20, algorithm67, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm68"; 24], Benchmarker.measure(20, algorithm68, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm69"; 24], Benchmarker.measure(20, algorithm69, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm70"; 24], Benchmarker.measure(20, algorithm70, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm71"; 24], Benchmarker.measure(20, algorithm71, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm72"; 24], Benchmarker.measure(20, algorithm72, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm73"; 24], Benchmarker.measure(20, algorithm73, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm74"; 24], Benchmarker.measure(20, algorithm74, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm75"; 24], Benchmarker.measure(20, algorithm75, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm76"; 24], Benchmarker.measure(20, algorithm76, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm77"; 24], Benchmarker.measure(20, algorithm77, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm78"; 24], Benchmarker.measure(20, algorithm78, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm79"; 24], Benchmarker.measure(20, algorithm79, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm80"; 24], Benchmarker.measure(20, algorithm80, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm81"; 24], Benchmarker.measure(20, algorithm81, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm82"; 24], Benchmarker.measure(20, algorithm82, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm83"; 24], Benchmarker.measure(20, algorithm83, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm84"; 24], Benchmarker.measure(20, algorithm84, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm85"; 24], Benchmarker.measure(20, algorithm85, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm86"; 24], Benchmarker.measure(20, algorithm86, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm87"; 24], Benchmarker.measure(20, algorithm87, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm88"; 24], Benchmarker.measure(20, algorithm88, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm89"; 24], Benchmarker.measure(20, algorithm89, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm90"; 24], Benchmarker.measure(20, algorithm90, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm91"; 24], Benchmarker.measure(20, algorithm91, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm92"; 24], Benchmarker.measure(20, algorithm92, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm93"; 24], Benchmarker.measure(20, algorithm93, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm94"; 24], Benchmarker.measure(20, algorithm94, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm95"; 24], Benchmarker.measure(20, algorithm95, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm96"; 24], Benchmarker.measure(20, algorithm96, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm97"; 24], Benchmarker.measure(20, algorithm97, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm98"; 24], Benchmarker.measure(20, algorithm98, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["algorithm99"; 24], Benchmarker.measure(20, algorithm99, map(MatrixGenerator.unwrap, matrices)...) );
    Benchmarker.add_data(plotter, ["naive_julia"; 24], Benchmarker.measure(20, naive, matrices...) );
    Benchmarker.add_data(plotter, ["recommended_julia"; 24], Benchmarker.measure(20, recommended, matrices...) );
    Benchmarker.finish(plotter);
    @info("Benchmarks complete")
end

main()
