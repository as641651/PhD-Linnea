using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm38(ml1268::Array{Float64,2}, ml1269::Array{Float64,2}, ml1270::Array{Float64,2}, ml1271::Array{Float64,2}, ml1272::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1268, full, L: ml1269, full, A: ml1270, full, B: ml1271, full, y: ml1272, full
    ml1273 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1273, ml1271)

    # R: ml1268, full, L: ml1269, full, A: ml1270, full, y: ml1272, full, tmp26: ml1273, full
    ml1274 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1270, ml1274, info) = LinearAlgebra.LAPACK.getrf!(ml1270)

    # R: ml1268, full, L: ml1269, full, y: ml1272, full, tmp26: ml1273, full, P11: ml1274, ipiv, L9: ml1270, lower_triangular_udiag, U10: ml1270, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1270, ml1273)

    # R: ml1268, full, L: ml1269, full, y: ml1272, full, P11: ml1274, ipiv, L9: ml1270, lower_triangular_udiag, tmp27: ml1273, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1270, ml1273)

    # R: ml1268, full, L: ml1269, full, y: ml1272, full, P11: ml1274, ipiv, tmp28: ml1273, full
    ml1275 = [1:length(ml1274);]
    @inbounds for i in 1:length(ml1274)
        ml1275[i], ml1275[ml1274[i]] = ml1275[ml1274[i]], ml1275[i];
    end;
    ml1276 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1276 = ml1273[invperm(ml1275),:]

    # R: ml1268, full, L: ml1269, full, y: ml1272, full, tmp25: ml1276, full
    ml1277 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1276, 0.0, ml1277)

    # R: ml1268, full, L: ml1269, full, y: ml1272, full, tmp19: ml1277, symmetric_lower_triangular
    ml1278 = diag(ml1269)
    ml1279 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1268, 1, ml1279, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1268, 2);
        view(ml1268, :, i)[:] .*= ml1278;
    end;        

    # R: ml1279, full, y: ml1272, full, tmp19: ml1277, symmetric_lower_triangular, tmp29: ml1268, full
    for i = 1:2000-1;
        view(ml1277, i, i+1:2000)[:] = view(ml1277, i+1:2000, i);
    end;
    ml1280 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1277, 1, ml1280, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1268, ml1279, 1.0, ml1277)

    # y: ml1272, full, tmp19: ml1280, full, tmp31: ml1277, full
    ml1281 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1277, ml1281, info) = LinearAlgebra.LAPACK.getrf!(ml1277)

    # y: ml1272, full, tmp19: ml1280, full, P35: ml1281, ipiv, L33: ml1277, lower_triangular_udiag, U34: ml1277, upper_triangular
    ml1282 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1280, ml1272, 0.0, ml1282)

    # P35: ml1281, ipiv, L33: ml1277, lower_triangular_udiag, U34: ml1277, upper_triangular, tmp32: ml1282, full
    ml1283 = [1:length(ml1281);]
    @inbounds for i in 1:length(ml1281)
        ml1283[i], ml1283[ml1281[i]] = ml1283[ml1281[i]], ml1283[i];
    end;
    ml1284 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1284 = ml1282[ml1283]

    # L33: ml1277, lower_triangular_udiag, U34: ml1277, upper_triangular, tmp40: ml1284, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1277, ml1284)

    # U34: ml1277, upper_triangular, tmp41: ml1284, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1277, ml1284)

    # tmp17: ml1284, full
    # x = tmp17
    return (ml1284)
end