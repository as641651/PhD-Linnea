using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm14(ml485::Array{Float64,2}, ml486::Array{Float64,2}, ml487::Array{Float64,2}, ml488::Array{Float64,2}, ml489::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml485, full, L: ml486, full, A: ml487, full, B: ml488, full, y: ml489, full
    ml490 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml490, ml488)

    # R: ml485, full, L: ml486, full, A: ml487, full, y: ml489, full, tmp26: ml490, full
    ml491 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml487, ml491, info) = LinearAlgebra.LAPACK.getrf!(ml487)

    # R: ml485, full, L: ml486, full, y: ml489, full, tmp26: ml490, full, P11: ml491, ipiv, L9: ml487, lower_triangular_udiag, U10: ml487, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml487, ml490)

    # R: ml485, full, L: ml486, full, y: ml489, full, P11: ml491, ipiv, L9: ml487, lower_triangular_udiag, tmp27: ml490, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml487, ml490)

    # R: ml485, full, L: ml486, full, y: ml489, full, P11: ml491, ipiv, tmp28: ml490, full
    ml492 = [1:length(ml491);]
    @inbounds for i in 1:length(ml491)
        ml492[i], ml492[ml491[i]] = ml492[ml491[i]], ml492[i];
    end;
    ml493 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml493 = ml490[invperm(ml492),:]

    # R: ml485, full, L: ml486, full, y: ml489, full, tmp25: ml493, full
    ml494 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml493, 0.0, ml494)

    # R: ml485, full, L: ml486, full, y: ml489, full, tmp19: ml494, symmetric_lower_triangular
    ml495 = diag(ml486)
    ml496 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml485, 1, ml496, 1)
    # tmp29 = (L R)
    for i = 1:size(ml485, 2);
        view(ml485, :, i)[:] .*= ml495;
    end;        

    # R: ml496, full, y: ml489, full, tmp19: ml494, symmetric_lower_triangular, tmp29: ml485, full
    for i = 1:2000-1;
        view(ml494, i, i+1:2000)[:] = view(ml494, i+1:2000, i);
    end;
    ml497 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml494, 1, ml497, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml485, ml496, 1.0, ml494)

    # y: ml489, full, tmp19: ml497, full, tmp31: ml494, full
    ml498 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml494, ml498, info) = LinearAlgebra.LAPACK.getrf!(ml494)

    # y: ml489, full, tmp19: ml497, full, P35: ml498, ipiv, L33: ml494, lower_triangular_udiag, U34: ml494, upper_triangular
    ml499 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml497, ml489, 0.0, ml499)

    # P35: ml498, ipiv, L33: ml494, lower_triangular_udiag, U34: ml494, upper_triangular, tmp32: ml499, full
    ml500 = [1:length(ml498);]
    @inbounds for i in 1:length(ml498)
        ml500[i], ml500[ml498[i]] = ml500[ml498[i]], ml500[i];
    end;
    ml501 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml501 = ml499[ml500]

    # L33: ml494, lower_triangular_udiag, U34: ml494, upper_triangular, tmp40: ml501, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml494, ml501)

    # U34: ml494, upper_triangular, tmp41: ml501, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml494, ml501)

    # tmp17: ml501, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml501), (finish-start)*1e-9)
end