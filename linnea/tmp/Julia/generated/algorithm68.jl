using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm68(ml2272::Array{Float64,2}, ml2273::Array{Float64,2}, ml2274::Array{Float64,2}, ml2275::Array{Float64,2}, ml2276::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2272, full, L: ml2273, full, A: ml2274, full, B: ml2275, full, y: ml2276, full
    ml2277 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2277, ml2275)

    # R: ml2272, full, L: ml2273, full, A: ml2274, full, y: ml2276, full, tmp26: ml2277, full
    ml2278 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2274, ml2278, info) = LinearAlgebra.LAPACK.getrf!(ml2274)

    # R: ml2272, full, L: ml2273, full, y: ml2276, full, tmp26: ml2277, full, P11: ml2278, ipiv, L9: ml2274, lower_triangular_udiag, U10: ml2274, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2274, ml2277)

    # R: ml2272, full, L: ml2273, full, y: ml2276, full, P11: ml2278, ipiv, L9: ml2274, lower_triangular_udiag, tmp27: ml2277, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2274, ml2277)

    # R: ml2272, full, L: ml2273, full, y: ml2276, full, P11: ml2278, ipiv, tmp28: ml2277, full
    ml2279 = [1:length(ml2278);]
    @inbounds for i in 1:length(ml2278)
        ml2279[i], ml2279[ml2278[i]] = ml2279[ml2278[i]], ml2279[i];
    end;
    ml2280 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2280 = ml2277[invperm(ml2279),:]

    # R: ml2272, full, L: ml2273, full, y: ml2276, full, tmp25: ml2280, full
    ml2281 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2280, 0.0, ml2281)

    # R: ml2272, full, L: ml2273, full, y: ml2276, full, tmp19: ml2281, symmetric_lower_triangular
    ml2282 = diag(ml2273)
    ml2283 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2272, 1, ml2283, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2272, 2);
        view(ml2272, :, i)[:] .*= ml2282;
    end;        

    # R: ml2283, full, y: ml2276, full, tmp19: ml2281, symmetric_lower_triangular, tmp29: ml2272, full
    for i = 1:2000-1;
        view(ml2281, i, i+1:2000)[:] = view(ml2281, i+1:2000, i);
    end;
    ml2284 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2281, 1, ml2284, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2272, ml2283, 1.0, ml2281)

    # y: ml2276, full, tmp19: ml2284, full, tmp31: ml2281, full
    ml2285 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2281, ml2285, info) = LinearAlgebra.LAPACK.getrf!(ml2281)

    # y: ml2276, full, tmp19: ml2284, full, P35: ml2285, ipiv, L33: ml2281, lower_triangular_udiag, U34: ml2281, upper_triangular
    ml2286 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2284, ml2276, 0.0, ml2286)

    # P35: ml2285, ipiv, L33: ml2281, lower_triangular_udiag, U34: ml2281, upper_triangular, tmp32: ml2286, full
    ml2287 = [1:length(ml2285);]
    @inbounds for i in 1:length(ml2285)
        ml2287[i], ml2287[ml2285[i]] = ml2287[ml2285[i]], ml2287[i];
    end;
    ml2288 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2288 = ml2286[ml2287]

    # L33: ml2281, lower_triangular_udiag, U34: ml2281, upper_triangular, tmp40: ml2288, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2281, ml2288)

    # U34: ml2281, upper_triangular, tmp41: ml2288, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2281, ml2288)

    # tmp17: ml2288, full
    # x = tmp17
    return (ml2288)
end