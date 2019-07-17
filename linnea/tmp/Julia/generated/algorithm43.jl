using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm43(ml1438::Array{Float64,2}, ml1439::Array{Float64,2}, ml1440::Array{Float64,2}, ml1441::Array{Float64,2}, ml1442::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1438, full, L: ml1439, full, A: ml1440, full, B: ml1441, full, y: ml1442, full
    ml1443 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1443, ml1441)

    # R: ml1438, full, L: ml1439, full, A: ml1440, full, y: ml1442, full, tmp26: ml1443, full
    ml1444 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1440, ml1444, info) = LinearAlgebra.LAPACK.getrf!(ml1440)

    # R: ml1438, full, L: ml1439, full, y: ml1442, full, tmp26: ml1443, full, P11: ml1444, ipiv, L9: ml1440, lower_triangular_udiag, U10: ml1440, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1440, ml1443)

    # R: ml1438, full, L: ml1439, full, y: ml1442, full, P11: ml1444, ipiv, L9: ml1440, lower_triangular_udiag, tmp27: ml1443, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1440, ml1443)

    # R: ml1438, full, L: ml1439, full, y: ml1442, full, P11: ml1444, ipiv, tmp28: ml1443, full
    ml1445 = [1:length(ml1444);]
    @inbounds for i in 1:length(ml1444)
        ml1445[i], ml1445[ml1444[i]] = ml1445[ml1444[i]], ml1445[i];
    end;
    ml1446 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1446 = ml1443[invperm(ml1445),:]

    # R: ml1438, full, L: ml1439, full, y: ml1442, full, tmp25: ml1446, full
    ml1447 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1446, 0.0, ml1447)

    # R: ml1438, full, L: ml1439, full, y: ml1442, full, tmp19: ml1447, symmetric_lower_triangular
    ml1448 = diag(ml1439)
    ml1449 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1438, 1, ml1449, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1438, 2);
        view(ml1438, :, i)[:] .*= ml1448;
    end;        

    # R: ml1449, full, y: ml1442, full, tmp19: ml1447, symmetric_lower_triangular, tmp29: ml1438, full
    for i = 1:2000-1;
        view(ml1447, i, i+1:2000)[:] = view(ml1447, i+1:2000, i);
    end;
    ml1450 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1447, 1, ml1450, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1438, ml1449, 1.0, ml1447)

    # y: ml1442, full, tmp19: ml1450, full, tmp31: ml1447, full
    ml1451 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1450, ml1442, 0.0, ml1451)

    # tmp31: ml1447, full, tmp32: ml1451, full
    ml1452 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1447, ml1452, info) = LinearAlgebra.LAPACK.getrf!(ml1447)

    # tmp32: ml1451, full, P35: ml1452, ipiv, L33: ml1447, lower_triangular_udiag, U34: ml1447, upper_triangular
    ml1453 = [1:length(ml1452);]
    @inbounds for i in 1:length(ml1452)
        ml1453[i], ml1453[ml1452[i]] = ml1453[ml1452[i]], ml1453[i];
    end;
    ml1454 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1454 = ml1451[ml1453]

    # L33: ml1447, lower_triangular_udiag, U34: ml1447, upper_triangular, tmp40: ml1454, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1447, ml1454)

    # U34: ml1447, upper_triangular, tmp41: ml1454, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1447, ml1454)

    # tmp17: ml1454, full
    # x = tmp17
    return (ml1454)
end