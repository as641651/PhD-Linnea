using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm81(ml2728::Array{Float64,2}, ml2729::Array{Float64,2}, ml2730::Array{Float64,2}, ml2731::Array{Float64,2}, ml2732::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2728, full, L: ml2729, full, A: ml2730, full, B: ml2731, full, y: ml2732, full
    ml2733 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2733, ml2731)

    # R: ml2728, full, L: ml2729, full, A: ml2730, full, y: ml2732, full, tmp26: ml2733, full
    ml2734 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2730, ml2734, info) = LinearAlgebra.LAPACK.getrf!(ml2730)

    # R: ml2728, full, L: ml2729, full, y: ml2732, full, tmp26: ml2733, full, P11: ml2734, ipiv, L9: ml2730, lower_triangular_udiag, U10: ml2730, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2730, ml2733)

    # R: ml2728, full, L: ml2729, full, y: ml2732, full, P11: ml2734, ipiv, L9: ml2730, lower_triangular_udiag, tmp27: ml2733, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2730, ml2733)

    # R: ml2728, full, L: ml2729, full, y: ml2732, full, P11: ml2734, ipiv, tmp28: ml2733, full
    ml2735 = [1:length(ml2734);]
    @inbounds for i in 1:length(ml2734)
        ml2735[i], ml2735[ml2734[i]] = ml2735[ml2734[i]], ml2735[i];
    end;
    ml2736 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2736 = ml2733[invperm(ml2735),:]

    # R: ml2728, full, L: ml2729, full, y: ml2732, full, tmp25: ml2736, full
    ml2737 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2736, 0.0, ml2737)

    # R: ml2728, full, L: ml2729, full, y: ml2732, full, tmp19: ml2737, symmetric_lower_triangular
    ml2738 = diag(ml2729)
    ml2739 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2728, 1, ml2739, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2728, 2);
        view(ml2728, :, i)[:] .*= ml2738;
    end;        

    # R: ml2739, full, y: ml2732, full, tmp19: ml2737, symmetric_lower_triangular, tmp29: ml2728, full
    ml2740 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2737, ml2732, 0.0, ml2740)

    # R: ml2739, full, tmp19: ml2737, symmetric_lower_triangular, tmp29: ml2728, full, tmp32: ml2740, full
    for i = 1:2000-1;
        view(ml2737, i, i+1:2000)[:] = view(ml2737, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2728, ml2739, 1.0, ml2737)

    # tmp32: ml2740, full, tmp31: ml2737, full
    ml2741 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2737, ml2741, info) = LinearAlgebra.LAPACK.getrf!(ml2737)

    # tmp32: ml2740, full, P35: ml2741, ipiv, L33: ml2737, lower_triangular_udiag, U34: ml2737, upper_triangular
    ml2742 = [1:length(ml2741);]
    @inbounds for i in 1:length(ml2741)
        ml2742[i], ml2742[ml2741[i]] = ml2742[ml2741[i]], ml2742[i];
    end;
    ml2743 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2743 = ml2740[ml2742]

    # L33: ml2737, lower_triangular_udiag, U34: ml2737, upper_triangular, tmp40: ml2743, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2737, ml2743)

    # U34: ml2737, upper_triangular, tmp41: ml2743, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2737, ml2743)

    # tmp17: ml2743, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2743), (finish-start)*1e-9)
end