using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm41(ml1370::Array{Float64,2}, ml1371::Array{Float64,2}, ml1372::Array{Float64,2}, ml1373::Array{Float64,2}, ml1374::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1370, full, L: ml1371, full, A: ml1372, full, B: ml1373, full, y: ml1374, full
    ml1375 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1375, ml1373)

    # R: ml1370, full, L: ml1371, full, A: ml1372, full, y: ml1374, full, tmp26: ml1375, full
    ml1376 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1372, ml1376, info) = LinearAlgebra.LAPACK.getrf!(ml1372)

    # R: ml1370, full, L: ml1371, full, y: ml1374, full, tmp26: ml1375, full, P11: ml1376, ipiv, L9: ml1372, lower_triangular_udiag, U10: ml1372, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1372, ml1375)

    # R: ml1370, full, L: ml1371, full, y: ml1374, full, P11: ml1376, ipiv, L9: ml1372, lower_triangular_udiag, tmp27: ml1375, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1372, ml1375)

    # R: ml1370, full, L: ml1371, full, y: ml1374, full, P11: ml1376, ipiv, tmp28: ml1375, full
    ml1377 = [1:length(ml1376);]
    @inbounds for i in 1:length(ml1376)
        ml1377[i], ml1377[ml1376[i]] = ml1377[ml1376[i]], ml1377[i];
    end;
    ml1378 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1378 = ml1375[invperm(ml1377),:]

    # R: ml1370, full, L: ml1371, full, y: ml1374, full, tmp25: ml1378, full
    ml1379 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1378, 0.0, ml1379)

    # R: ml1370, full, L: ml1371, full, y: ml1374, full, tmp19: ml1379, symmetric_lower_triangular
    ml1380 = diag(ml1371)
    ml1381 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1370, 1, ml1381, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1370, 2);
        view(ml1370, :, i)[:] .*= ml1380;
    end;        

    # R: ml1381, full, y: ml1374, full, tmp19: ml1379, symmetric_lower_triangular, tmp29: ml1370, full
    for i = 1:2000-1;
        view(ml1379, i, i+1:2000)[:] = view(ml1379, i+1:2000, i);
    end;
    ml1382 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1379, 1, ml1382, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1370, ml1381, 1.0, ml1379)

    # y: ml1374, full, tmp19: ml1382, full, tmp31: ml1379, full
    ml1383 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1382, ml1374, 0.0, ml1383)

    # tmp31: ml1379, full, tmp32: ml1383, full
    ml1384 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1379, ml1384, info) = LinearAlgebra.LAPACK.getrf!(ml1379)

    # tmp32: ml1383, full, P35: ml1384, ipiv, L33: ml1379, lower_triangular_udiag, U34: ml1379, upper_triangular
    ml1385 = [1:length(ml1384);]
    @inbounds for i in 1:length(ml1384)
        ml1385[i], ml1385[ml1384[i]] = ml1385[ml1384[i]], ml1385[i];
    end;
    ml1386 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1386 = ml1383[ml1385]

    # L33: ml1379, lower_triangular_udiag, U34: ml1379, upper_triangular, tmp40: ml1386, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1379, ml1386)

    # U34: ml1379, upper_triangular, tmp41: ml1386, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1379, ml1386)

    # tmp17: ml1386, full
    # x = tmp17
    return (ml1386)
end