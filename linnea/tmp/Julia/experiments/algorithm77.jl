using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm77(ml2595::Array{Float64,2}, ml2596::Array{Float64,2}, ml2597::Array{Float64,2}, ml2598::Array{Float64,2}, ml2599::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2595, full, L: ml2596, full, A: ml2597, full, B: ml2598, full, y: ml2599, full
    ml2600 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2600, ml2598)

    # R: ml2595, full, L: ml2596, full, A: ml2597, full, y: ml2599, full, tmp26: ml2600, full
    ml2601 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2597, ml2601, info) = LinearAlgebra.LAPACK.getrf!(ml2597)

    # R: ml2595, full, L: ml2596, full, y: ml2599, full, tmp26: ml2600, full, P11: ml2601, ipiv, L9: ml2597, lower_triangular_udiag, U10: ml2597, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2597, ml2600)

    # R: ml2595, full, L: ml2596, full, y: ml2599, full, P11: ml2601, ipiv, L9: ml2597, lower_triangular_udiag, tmp27: ml2600, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2597, ml2600)

    # R: ml2595, full, L: ml2596, full, y: ml2599, full, P11: ml2601, ipiv, tmp28: ml2600, full
    ml2602 = [1:length(ml2601);]
    @inbounds for i in 1:length(ml2601)
        ml2602[i], ml2602[ml2601[i]] = ml2602[ml2601[i]], ml2602[i];
    end;
    ml2603 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2603 = ml2600[invperm(ml2602),:]

    # R: ml2595, full, L: ml2596, full, y: ml2599, full, tmp25: ml2603, full
    ml2604 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2603, 0.0, ml2604)

    # R: ml2595, full, L: ml2596, full, y: ml2599, full, tmp19: ml2604, symmetric_lower_triangular
    ml2605 = diag(ml2596)
    ml2606 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2595, 1, ml2606, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2595, 2);
        view(ml2595, :, i)[:] .*= ml2605;
    end;        

    # R: ml2606, full, y: ml2599, full, tmp19: ml2604, symmetric_lower_triangular, tmp29: ml2595, full
    for i = 1:2000-1;
        view(ml2604, i, i+1:2000)[:] = view(ml2604, i+1:2000, i);
    end;
    ml2607 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2604, 1, ml2607, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2595, ml2606, 1.0, ml2604)

    # y: ml2599, full, tmp19: ml2607, full, tmp31: ml2604, full
    ml2608 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2604, ml2608, info) = LinearAlgebra.LAPACK.getrf!(ml2604)

    # y: ml2599, full, tmp19: ml2607, full, P35: ml2608, ipiv, L33: ml2604, lower_triangular_udiag, U34: ml2604, upper_triangular
    ml2609 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2607, ml2599, 0.0, ml2609)

    # P35: ml2608, ipiv, L33: ml2604, lower_triangular_udiag, U34: ml2604, upper_triangular, tmp32: ml2609, full
    ml2610 = [1:length(ml2608);]
    @inbounds for i in 1:length(ml2608)
        ml2610[i], ml2610[ml2608[i]] = ml2610[ml2608[i]], ml2610[i];
    end;
    ml2611 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2611 = ml2609[ml2610]

    # L33: ml2604, lower_triangular_udiag, U34: ml2604, upper_triangular, tmp40: ml2611, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2604, ml2611)

    # U34: ml2604, upper_triangular, tmp41: ml2611, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2604, ml2611)

    # tmp17: ml2611, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2611), (finish-start)*1e-9)
end