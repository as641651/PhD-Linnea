using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm9(ml315::Array{Float64,2}, ml316::Array{Float64,2}, ml317::Array{Float64,2}, ml318::Array{Float64,2}, ml319::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml315, full, L: ml316, full, A: ml317, full, B: ml318, full, y: ml319, full
    ml320 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml320, ml318)

    # R: ml315, full, L: ml316, full, A: ml317, full, y: ml319, full, tmp26: ml320, full
    ml321 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml317, ml321, info) = LinearAlgebra.LAPACK.getrf!(ml317)

    # R: ml315, full, L: ml316, full, y: ml319, full, tmp26: ml320, full, P11: ml321, ipiv, L9: ml317, lower_triangular_udiag, U10: ml317, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml317, ml320)

    # R: ml315, full, L: ml316, full, y: ml319, full, P11: ml321, ipiv, L9: ml317, lower_triangular_udiag, tmp27: ml320, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml317, ml320)

    # R: ml315, full, L: ml316, full, y: ml319, full, P11: ml321, ipiv, tmp28: ml320, full
    ml322 = [1:length(ml321);]
    @inbounds for i in 1:length(ml321)
        ml322[i], ml322[ml321[i]] = ml322[ml321[i]], ml322[i];
    end;
    ml323 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml323 = ml320[invperm(ml322),:]

    # R: ml315, full, L: ml316, full, y: ml319, full, tmp25: ml323, full
    ml324 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml323, 0.0, ml324)

    # R: ml315, full, L: ml316, full, y: ml319, full, tmp19: ml324, symmetric_lower_triangular
    ml325 = diag(ml316)
    ml326 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml315, 1, ml326, 1)
    # tmp29 = (L R)
    for i = 1:size(ml315, 2);
        view(ml315, :, i)[:] .*= ml325;
    end;        

    # R: ml326, full, y: ml319, full, tmp19: ml324, symmetric_lower_triangular, tmp29: ml315, full
    for i = 1:2000-1;
        view(ml324, i, i+1:2000)[:] = view(ml324, i+1:2000, i);
    end;
    ml327 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml324, 1, ml327, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml315, ml326, 1.0, ml324)

    # y: ml319, full, tmp19: ml327, full, tmp31: ml324, full
    ml328 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml324, ml328, info) = LinearAlgebra.LAPACK.getrf!(ml324)

    # y: ml319, full, tmp19: ml327, full, P35: ml328, ipiv, L33: ml324, lower_triangular_udiag, U34: ml324, upper_triangular
    ml329 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml327, ml319, 0.0, ml329)

    # P35: ml328, ipiv, L33: ml324, lower_triangular_udiag, U34: ml324, upper_triangular, tmp32: ml329, full
    ml330 = [1:length(ml328);]
    @inbounds for i in 1:length(ml328)
        ml330[i], ml330[ml328[i]] = ml330[ml328[i]], ml330[i];
    end;
    ml331 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml331 = ml329[ml330]

    # L33: ml324, lower_triangular_udiag, U34: ml324, upper_triangular, tmp40: ml331, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml324, ml331)

    # U34: ml324, upper_triangular, tmp41: ml331, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml324, ml331)

    # tmp17: ml331, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml331), (finish-start)*1e-9)
end