using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm80(ml2696::Array{Float64,2}, ml2697::Array{Float64,2}, ml2698::Array{Float64,2}, ml2699::Array{Float64,2}, ml2700::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2696, full, L: ml2697, full, A: ml2698, full, B: ml2699, full, y: ml2700, full
    ml2701 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2701, ml2699)

    # R: ml2696, full, L: ml2697, full, A: ml2698, full, y: ml2700, full, tmp26: ml2701, full
    ml2702 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2698, ml2702, info) = LinearAlgebra.LAPACK.getrf!(ml2698)

    # R: ml2696, full, L: ml2697, full, y: ml2700, full, tmp26: ml2701, full, P11: ml2702, ipiv, L9: ml2698, lower_triangular_udiag, U10: ml2698, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2698, ml2701)

    # R: ml2696, full, L: ml2697, full, y: ml2700, full, P11: ml2702, ipiv, L9: ml2698, lower_triangular_udiag, tmp27: ml2701, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2698, ml2701)

    # R: ml2696, full, L: ml2697, full, y: ml2700, full, P11: ml2702, ipiv, tmp28: ml2701, full
    ml2703 = [1:length(ml2702);]
    @inbounds for i in 1:length(ml2702)
        ml2703[i], ml2703[ml2702[i]] = ml2703[ml2702[i]], ml2703[i];
    end;
    ml2704 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2704 = ml2701[invperm(ml2703),:]

    # R: ml2696, full, L: ml2697, full, y: ml2700, full, tmp25: ml2704, full
    ml2705 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2704, 0.0, ml2705)

    # R: ml2696, full, L: ml2697, full, y: ml2700, full, tmp19: ml2705, symmetric_lower_triangular
    ml2706 = diag(ml2697)
    ml2707 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2696, 1, ml2707, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2696, 2);
        view(ml2696, :, i)[:] .*= ml2706;
    end;        

    # R: ml2707, full, y: ml2700, full, tmp19: ml2705, symmetric_lower_triangular, tmp29: ml2696, full
    ml2708 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2705, ml2700, 0.0, ml2708)

    # R: ml2707, full, tmp19: ml2705, symmetric_lower_triangular, tmp29: ml2696, full, tmp32: ml2708, full
    for i = 1:2000-1;
        view(ml2705, i, i+1:2000)[:] = view(ml2705, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2696, ml2707, 1.0, ml2705)

    # tmp32: ml2708, full, tmp31: ml2705, full
    ml2709 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2705, ml2709, info) = LinearAlgebra.LAPACK.getrf!(ml2705)

    # tmp32: ml2708, full, P35: ml2709, ipiv, L33: ml2705, lower_triangular_udiag, U34: ml2705, upper_triangular
    ml2710 = [1:length(ml2709);]
    @inbounds for i in 1:length(ml2709)
        ml2710[i], ml2710[ml2709[i]] = ml2710[ml2709[i]], ml2710[i];
    end;
    ml2711 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2711 = ml2708[ml2710]

    # L33: ml2705, lower_triangular_udiag, U34: ml2705, upper_triangular, tmp40: ml2711, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2705, ml2711)

    # U34: ml2705, upper_triangular, tmp41: ml2711, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2705, ml2711)

    # tmp17: ml2711, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2711), (finish-start)*1e-9)
end