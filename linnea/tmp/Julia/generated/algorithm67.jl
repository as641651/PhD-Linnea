using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm67(ml2238::Array{Float64,2}, ml2239::Array{Float64,2}, ml2240::Array{Float64,2}, ml2241::Array{Float64,2}, ml2242::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2238, full, L: ml2239, full, A: ml2240, full, B: ml2241, full, y: ml2242, full
    ml2243 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2243, ml2241)

    # R: ml2238, full, L: ml2239, full, A: ml2240, full, y: ml2242, full, tmp26: ml2243, full
    ml2244 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2240, ml2244, info) = LinearAlgebra.LAPACK.getrf!(ml2240)

    # R: ml2238, full, L: ml2239, full, y: ml2242, full, tmp26: ml2243, full, P11: ml2244, ipiv, L9: ml2240, lower_triangular_udiag, U10: ml2240, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2240, ml2243)

    # R: ml2238, full, L: ml2239, full, y: ml2242, full, P11: ml2244, ipiv, L9: ml2240, lower_triangular_udiag, tmp27: ml2243, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2240, ml2243)

    # R: ml2238, full, L: ml2239, full, y: ml2242, full, P11: ml2244, ipiv, tmp28: ml2243, full
    ml2245 = [1:length(ml2244);]
    @inbounds for i in 1:length(ml2244)
        ml2245[i], ml2245[ml2244[i]] = ml2245[ml2244[i]], ml2245[i];
    end;
    ml2246 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2246 = ml2243[invperm(ml2245),:]

    # R: ml2238, full, L: ml2239, full, y: ml2242, full, tmp25: ml2246, full
    ml2247 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2246, 0.0, ml2247)

    # R: ml2238, full, L: ml2239, full, y: ml2242, full, tmp19: ml2247, symmetric_lower_triangular
    ml2248 = diag(ml2239)
    ml2249 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2238, 1, ml2249, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2238, 2);
        view(ml2238, :, i)[:] .*= ml2248;
    end;        

    # R: ml2249, full, y: ml2242, full, tmp19: ml2247, symmetric_lower_triangular, tmp29: ml2238, full
    for i = 1:2000-1;
        view(ml2247, i, i+1:2000)[:] = view(ml2247, i+1:2000, i);
    end;
    ml2250 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2247, 1, ml2250, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2238, ml2249, 1.0, ml2247)

    # y: ml2242, full, tmp19: ml2250, full, tmp31: ml2247, full
    ml2251 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2247, ml2251, info) = LinearAlgebra.LAPACK.getrf!(ml2247)

    # y: ml2242, full, tmp19: ml2250, full, P35: ml2251, ipiv, L33: ml2247, lower_triangular_udiag, U34: ml2247, upper_triangular
    ml2252 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2250, ml2242, 0.0, ml2252)

    # P35: ml2251, ipiv, L33: ml2247, lower_triangular_udiag, U34: ml2247, upper_triangular, tmp32: ml2252, full
    ml2253 = [1:length(ml2251);]
    @inbounds for i in 1:length(ml2251)
        ml2253[i], ml2253[ml2251[i]] = ml2253[ml2251[i]], ml2253[i];
    end;
    ml2254 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2254 = ml2252[ml2253]

    # L33: ml2247, lower_triangular_udiag, U34: ml2247, upper_triangular, tmp40: ml2254, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2247, ml2254)

    # U34: ml2247, upper_triangular, tmp41: ml2254, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2247, ml2254)

    # tmp17: ml2254, full
    # x = tmp17
    return (ml2254)
end