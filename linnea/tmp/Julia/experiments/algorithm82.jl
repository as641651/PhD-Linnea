using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm82(ml2760::Array{Float64,2}, ml2761::Array{Float64,2}, ml2762::Array{Float64,2}, ml2763::Array{Float64,2}, ml2764::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2760, full, L: ml2761, full, A: ml2762, full, B: ml2763, full, y: ml2764, full
    ml2765 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2765, ml2763)

    # R: ml2760, full, L: ml2761, full, A: ml2762, full, y: ml2764, full, tmp26: ml2765, full
    ml2766 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2762, ml2766, info) = LinearAlgebra.LAPACK.getrf!(ml2762)

    # R: ml2760, full, L: ml2761, full, y: ml2764, full, tmp26: ml2765, full, P11: ml2766, ipiv, L9: ml2762, lower_triangular_udiag, U10: ml2762, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2762, ml2765)

    # R: ml2760, full, L: ml2761, full, y: ml2764, full, P11: ml2766, ipiv, L9: ml2762, lower_triangular_udiag, tmp27: ml2765, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2762, ml2765)

    # R: ml2760, full, L: ml2761, full, y: ml2764, full, P11: ml2766, ipiv, tmp28: ml2765, full
    ml2767 = [1:length(ml2766);]
    @inbounds for i in 1:length(ml2766)
        ml2767[i], ml2767[ml2766[i]] = ml2767[ml2766[i]], ml2767[i];
    end;
    ml2768 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2768 = ml2765[invperm(ml2767),:]

    # R: ml2760, full, L: ml2761, full, y: ml2764, full, tmp25: ml2768, full
    ml2769 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2768, 0.0, ml2769)

    # R: ml2760, full, L: ml2761, full, y: ml2764, full, tmp19: ml2769, symmetric_lower_triangular
    ml2770 = diag(ml2761)
    ml2771 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2760, 1, ml2771, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2760, 2);
        view(ml2760, :, i)[:] .*= ml2770;
    end;        

    # R: ml2771, full, y: ml2764, full, tmp19: ml2769, symmetric_lower_triangular, tmp29: ml2760, full
    ml2772 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2769, ml2764, 0.0, ml2772)

    # R: ml2771, full, tmp19: ml2769, symmetric_lower_triangular, tmp29: ml2760, full, tmp32: ml2772, full
    for i = 1:2000-1;
        view(ml2769, i, i+1:2000)[:] = view(ml2769, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2760, ml2771, 1.0, ml2769)

    # tmp32: ml2772, full, tmp31: ml2769, full
    ml2773 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2769, ml2773, info) = LinearAlgebra.LAPACK.getrf!(ml2769)

    # tmp32: ml2772, full, P35: ml2773, ipiv, L33: ml2769, lower_triangular_udiag, U34: ml2769, upper_triangular
    ml2774 = [1:length(ml2773);]
    @inbounds for i in 1:length(ml2773)
        ml2774[i], ml2774[ml2773[i]] = ml2774[ml2773[i]], ml2774[i];
    end;
    ml2775 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2775 = ml2772[ml2774]

    # L33: ml2769, lower_triangular_udiag, U34: ml2769, upper_triangular, tmp40: ml2775, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2769, ml2775)

    # U34: ml2769, upper_triangular, tmp41: ml2775, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2769, ml2775)

    # tmp17: ml2775, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2775), (finish-start)*1e-9)
end