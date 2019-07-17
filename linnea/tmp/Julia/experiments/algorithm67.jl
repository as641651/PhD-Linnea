using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm67(ml2255::Array{Float64,2}, ml2256::Array{Float64,2}, ml2257::Array{Float64,2}, ml2258::Array{Float64,2}, ml2259::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2255, full, L: ml2256, full, A: ml2257, full, B: ml2258, full, y: ml2259, full
    ml2260 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2260, ml2258)

    # R: ml2255, full, L: ml2256, full, A: ml2257, full, y: ml2259, full, tmp26: ml2260, full
    ml2261 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2257, ml2261, info) = LinearAlgebra.LAPACK.getrf!(ml2257)

    # R: ml2255, full, L: ml2256, full, y: ml2259, full, tmp26: ml2260, full, P11: ml2261, ipiv, L9: ml2257, lower_triangular_udiag, U10: ml2257, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2257, ml2260)

    # R: ml2255, full, L: ml2256, full, y: ml2259, full, P11: ml2261, ipiv, L9: ml2257, lower_triangular_udiag, tmp27: ml2260, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2257, ml2260)

    # R: ml2255, full, L: ml2256, full, y: ml2259, full, P11: ml2261, ipiv, tmp28: ml2260, full
    ml2262 = [1:length(ml2261);]
    @inbounds for i in 1:length(ml2261)
        ml2262[i], ml2262[ml2261[i]] = ml2262[ml2261[i]], ml2262[i];
    end;
    ml2263 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2263 = ml2260[invperm(ml2262),:]

    # R: ml2255, full, L: ml2256, full, y: ml2259, full, tmp25: ml2263, full
    ml2264 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2263, 0.0, ml2264)

    # R: ml2255, full, L: ml2256, full, y: ml2259, full, tmp19: ml2264, symmetric_lower_triangular
    ml2265 = diag(ml2256)
    ml2266 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2255, 1, ml2266, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2255, 2);
        view(ml2255, :, i)[:] .*= ml2265;
    end;        

    # R: ml2266, full, y: ml2259, full, tmp19: ml2264, symmetric_lower_triangular, tmp29: ml2255, full
    for i = 1:2000-1;
        view(ml2264, i, i+1:2000)[:] = view(ml2264, i+1:2000, i);
    end;
    ml2267 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2264, 1, ml2267, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2255, ml2266, 1.0, ml2264)

    # y: ml2259, full, tmp19: ml2267, full, tmp31: ml2264, full
    ml2268 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2264, ml2268, info) = LinearAlgebra.LAPACK.getrf!(ml2264)

    # y: ml2259, full, tmp19: ml2267, full, P35: ml2268, ipiv, L33: ml2264, lower_triangular_udiag, U34: ml2264, upper_triangular
    ml2269 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2267, ml2259, 0.0, ml2269)

    # P35: ml2268, ipiv, L33: ml2264, lower_triangular_udiag, U34: ml2264, upper_triangular, tmp32: ml2269, full
    ml2270 = [1:length(ml2268);]
    @inbounds for i in 1:length(ml2268)
        ml2270[i], ml2270[ml2268[i]] = ml2270[ml2268[i]], ml2270[i];
    end;
    ml2271 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2271 = ml2269[ml2270]

    # L33: ml2264, lower_triangular_udiag, U34: ml2264, upper_triangular, tmp40: ml2271, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2264, ml2271)

    # U34: ml2264, upper_triangular, tmp41: ml2271, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2264, ml2271)

    # tmp17: ml2271, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2271), (finish-start)*1e-9)
end