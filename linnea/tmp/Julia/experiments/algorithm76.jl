using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm76(ml2561::Array{Float64,2}, ml2562::Array{Float64,2}, ml2563::Array{Float64,2}, ml2564::Array{Float64,2}, ml2565::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2561, full, L: ml2562, full, A: ml2563, full, B: ml2564, full, y: ml2565, full
    ml2566 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2566, ml2564)

    # R: ml2561, full, L: ml2562, full, A: ml2563, full, y: ml2565, full, tmp26: ml2566, full
    ml2567 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2563, ml2567, info) = LinearAlgebra.LAPACK.getrf!(ml2563)

    # R: ml2561, full, L: ml2562, full, y: ml2565, full, tmp26: ml2566, full, P11: ml2567, ipiv, L9: ml2563, lower_triangular_udiag, U10: ml2563, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2563, ml2566)

    # R: ml2561, full, L: ml2562, full, y: ml2565, full, P11: ml2567, ipiv, L9: ml2563, lower_triangular_udiag, tmp27: ml2566, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2563, ml2566)

    # R: ml2561, full, L: ml2562, full, y: ml2565, full, P11: ml2567, ipiv, tmp28: ml2566, full
    ml2568 = [1:length(ml2567);]
    @inbounds for i in 1:length(ml2567)
        ml2568[i], ml2568[ml2567[i]] = ml2568[ml2567[i]], ml2568[i];
    end;
    ml2569 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2569 = ml2566[invperm(ml2568),:]

    # R: ml2561, full, L: ml2562, full, y: ml2565, full, tmp25: ml2569, full
    ml2570 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2569, 0.0, ml2570)

    # R: ml2561, full, L: ml2562, full, y: ml2565, full, tmp19: ml2570, symmetric_lower_triangular
    ml2571 = diag(ml2562)
    ml2572 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2561, 1, ml2572, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2561, 2);
        view(ml2561, :, i)[:] .*= ml2571;
    end;        

    # R: ml2572, full, y: ml2565, full, tmp19: ml2570, symmetric_lower_triangular, tmp29: ml2561, full
    for i = 1:2000-1;
        view(ml2570, i, i+1:2000)[:] = view(ml2570, i+1:2000, i);
    end;
    ml2573 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2570, 1, ml2573, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2561, ml2572, 1.0, ml2570)

    # y: ml2565, full, tmp19: ml2573, full, tmp31: ml2570, full
    ml2574 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2570, ml2574, info) = LinearAlgebra.LAPACK.getrf!(ml2570)

    # y: ml2565, full, tmp19: ml2573, full, P35: ml2574, ipiv, L33: ml2570, lower_triangular_udiag, U34: ml2570, upper_triangular
    ml2575 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2573, ml2565, 0.0, ml2575)

    # P35: ml2574, ipiv, L33: ml2570, lower_triangular_udiag, U34: ml2570, upper_triangular, tmp32: ml2575, full
    ml2576 = [1:length(ml2574);]
    @inbounds for i in 1:length(ml2574)
        ml2576[i], ml2576[ml2574[i]] = ml2576[ml2574[i]], ml2576[i];
    end;
    ml2577 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2577 = ml2575[ml2576]

    # L33: ml2570, lower_triangular_udiag, U34: ml2570, upper_triangular, tmp40: ml2577, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2570, ml2577)

    # U34: ml2570, upper_triangular, tmp41: ml2577, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2570, ml2577)

    # tmp17: ml2577, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2577), (finish-start)*1e-9)
end