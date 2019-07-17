using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm97(ml3259::Array{Float64,2}, ml3260::Array{Float64,2}, ml3261::Array{Float64,2}, ml3262::Array{Float64,2}, ml3263::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3259, full, L: ml3260, full, A: ml3261, full, B: ml3262, full, y: ml3263, full
    ml3264 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3264, ml3262)

    # R: ml3259, full, L: ml3260, full, A: ml3261, full, y: ml3263, full, tmp26: ml3264, full
    ml3265 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3261, ml3265, info) = LinearAlgebra.LAPACK.getrf!(ml3261)

    # R: ml3259, full, L: ml3260, full, y: ml3263, full, tmp26: ml3264, full, P11: ml3265, ipiv, L9: ml3261, lower_triangular_udiag, U10: ml3261, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3261, ml3264)

    # R: ml3259, full, L: ml3260, full, y: ml3263, full, P11: ml3265, ipiv, L9: ml3261, lower_triangular_udiag, tmp27: ml3264, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3261, ml3264)

    # R: ml3259, full, L: ml3260, full, y: ml3263, full, P11: ml3265, ipiv, tmp28: ml3264, full
    ml3266 = [1:length(ml3265);]
    @inbounds for i in 1:length(ml3265)
        ml3266[i], ml3266[ml3265[i]] = ml3266[ml3265[i]], ml3266[i];
    end;
    ml3267 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3267 = ml3264[invperm(ml3266),:]

    # R: ml3259, full, L: ml3260, full, y: ml3263, full, tmp25: ml3267, full
    ml3268 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3267, 0.0, ml3268)

    # R: ml3259, full, L: ml3260, full, y: ml3263, full, tmp19: ml3268, symmetric_lower_triangular
    ml3269 = diag(ml3260)
    ml3270 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3259, 1, ml3270, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3259, 2);
        view(ml3259, :, i)[:] .*= ml3269;
    end;        

    # R: ml3270, full, y: ml3263, full, tmp19: ml3268, symmetric_lower_triangular, tmp29: ml3259, full
    for i = 1:2000-1;
        view(ml3268, i, i+1:2000)[:] = view(ml3268, i+1:2000, i);
    end;
    ml3271 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3268, 1, ml3271, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3259, ml3270, 1.0, ml3268)

    # y: ml3263, full, tmp19: ml3271, full, tmp31: ml3268, full
    ml3272 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3268, ml3272, info) = LinearAlgebra.LAPACK.getrf!(ml3268)

    # y: ml3263, full, tmp19: ml3271, full, P35: ml3272, ipiv, L33: ml3268, lower_triangular_udiag, U34: ml3268, upper_triangular
    ml3273 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3271, ml3263, 0.0, ml3273)

    # P35: ml3272, ipiv, L33: ml3268, lower_triangular_udiag, U34: ml3268, upper_triangular, tmp32: ml3273, full
    ml3274 = [1:length(ml3272);]
    @inbounds for i in 1:length(ml3272)
        ml3274[i], ml3274[ml3272[i]] = ml3274[ml3272[i]], ml3274[i];
    end;
    ml3275 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3275 = ml3273[ml3274]

    # L33: ml3268, lower_triangular_udiag, U34: ml3268, upper_triangular, tmp40: ml3275, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3268, ml3275)

    # U34: ml3268, upper_triangular, tmp41: ml3275, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3268, ml3275)

    # tmp17: ml3275, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3275), (finish-start)*1e-9)
end