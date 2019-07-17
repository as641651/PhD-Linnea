using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm75(ml2527::Array{Float64,2}, ml2528::Array{Float64,2}, ml2529::Array{Float64,2}, ml2530::Array{Float64,2}, ml2531::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2527, full, L: ml2528, full, A: ml2529, full, B: ml2530, full, y: ml2531, full
    ml2532 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2532, ml2530)

    # R: ml2527, full, L: ml2528, full, A: ml2529, full, y: ml2531, full, tmp26: ml2532, full
    ml2533 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2529, ml2533, info) = LinearAlgebra.LAPACK.getrf!(ml2529)

    # R: ml2527, full, L: ml2528, full, y: ml2531, full, tmp26: ml2532, full, P11: ml2533, ipiv, L9: ml2529, lower_triangular_udiag, U10: ml2529, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2529, ml2532)

    # R: ml2527, full, L: ml2528, full, y: ml2531, full, P11: ml2533, ipiv, L9: ml2529, lower_triangular_udiag, tmp27: ml2532, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2529, ml2532)

    # R: ml2527, full, L: ml2528, full, y: ml2531, full, P11: ml2533, ipiv, tmp28: ml2532, full
    ml2534 = [1:length(ml2533);]
    @inbounds for i in 1:length(ml2533)
        ml2534[i], ml2534[ml2533[i]] = ml2534[ml2533[i]], ml2534[i];
    end;
    ml2535 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2535 = ml2532[invperm(ml2534),:]

    # R: ml2527, full, L: ml2528, full, y: ml2531, full, tmp25: ml2535, full
    ml2536 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2535, 0.0, ml2536)

    # R: ml2527, full, L: ml2528, full, y: ml2531, full, tmp19: ml2536, symmetric_lower_triangular
    ml2537 = diag(ml2528)
    ml2538 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2527, 1, ml2538, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2527, 2);
        view(ml2527, :, i)[:] .*= ml2537;
    end;        

    # R: ml2538, full, y: ml2531, full, tmp19: ml2536, symmetric_lower_triangular, tmp29: ml2527, full
    for i = 1:2000-1;
        view(ml2536, i, i+1:2000)[:] = view(ml2536, i+1:2000, i);
    end;
    ml2539 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2536, 1, ml2539, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2527, ml2538, 1.0, ml2536)

    # y: ml2531, full, tmp19: ml2539, full, tmp31: ml2536, full
    ml2540 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2536, ml2540, info) = LinearAlgebra.LAPACK.getrf!(ml2536)

    # y: ml2531, full, tmp19: ml2539, full, P35: ml2540, ipiv, L33: ml2536, lower_triangular_udiag, U34: ml2536, upper_triangular
    ml2541 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2539, ml2531, 0.0, ml2541)

    # P35: ml2540, ipiv, L33: ml2536, lower_triangular_udiag, U34: ml2536, upper_triangular, tmp32: ml2541, full
    ml2542 = [1:length(ml2540);]
    @inbounds for i in 1:length(ml2540)
        ml2542[i], ml2542[ml2540[i]] = ml2542[ml2540[i]], ml2542[i];
    end;
    ml2543 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2543 = ml2541[ml2542]

    # L33: ml2536, lower_triangular_udiag, U34: ml2536, upper_triangular, tmp40: ml2543, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2536, ml2543)

    # U34: ml2536, upper_triangular, tmp41: ml2543, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2536, ml2543)

    # tmp17: ml2543, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2543), (finish-start)*1e-9)
end