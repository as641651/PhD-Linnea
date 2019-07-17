using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm44(ml1472::Array{Float64,2}, ml1473::Array{Float64,2}, ml1474::Array{Float64,2}, ml1475::Array{Float64,2}, ml1476::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1472, full, L: ml1473, full, A: ml1474, full, B: ml1475, full, y: ml1476, full
    ml1477 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1477, ml1475)

    # R: ml1472, full, L: ml1473, full, A: ml1474, full, y: ml1476, full, tmp26: ml1477, full
    ml1478 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1474, ml1478, info) = LinearAlgebra.LAPACK.getrf!(ml1474)

    # R: ml1472, full, L: ml1473, full, y: ml1476, full, tmp26: ml1477, full, P11: ml1478, ipiv, L9: ml1474, lower_triangular_udiag, U10: ml1474, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1474, ml1477)

    # R: ml1472, full, L: ml1473, full, y: ml1476, full, P11: ml1478, ipiv, L9: ml1474, lower_triangular_udiag, tmp27: ml1477, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1474, ml1477)

    # R: ml1472, full, L: ml1473, full, y: ml1476, full, P11: ml1478, ipiv, tmp28: ml1477, full
    ml1479 = [1:length(ml1478);]
    @inbounds for i in 1:length(ml1478)
        ml1479[i], ml1479[ml1478[i]] = ml1479[ml1478[i]], ml1479[i];
    end;
    ml1480 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1480 = ml1477[invperm(ml1479),:]

    # R: ml1472, full, L: ml1473, full, y: ml1476, full, tmp25: ml1480, full
    ml1481 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1480, 0.0, ml1481)

    # R: ml1472, full, L: ml1473, full, y: ml1476, full, tmp19: ml1481, symmetric_lower_triangular
    ml1482 = diag(ml1473)
    ml1483 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1472, 1, ml1483, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1472, 2);
        view(ml1472, :, i)[:] .*= ml1482;
    end;        

    # R: ml1483, full, y: ml1476, full, tmp19: ml1481, symmetric_lower_triangular, tmp29: ml1472, full
    for i = 1:2000-1;
        view(ml1481, i, i+1:2000)[:] = view(ml1481, i+1:2000, i);
    end;
    ml1484 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1481, 1, ml1484, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1472, ml1483, 1.0, ml1481)

    # y: ml1476, full, tmp19: ml1484, full, tmp31: ml1481, full
    ml1485 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1481, ml1485, info) = LinearAlgebra.LAPACK.getrf!(ml1481)

    # y: ml1476, full, tmp19: ml1484, full, P35: ml1485, ipiv, L33: ml1481, lower_triangular_udiag, U34: ml1481, upper_triangular
    ml1486 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1484, ml1476, 0.0, ml1486)

    # P35: ml1485, ipiv, L33: ml1481, lower_triangular_udiag, U34: ml1481, upper_triangular, tmp32: ml1486, full
    ml1487 = [1:length(ml1485);]
    @inbounds for i in 1:length(ml1485)
        ml1487[i], ml1487[ml1485[i]] = ml1487[ml1485[i]], ml1487[i];
    end;
    ml1488 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1488 = ml1486[ml1487]

    # L33: ml1481, lower_triangular_udiag, U34: ml1481, upper_triangular, tmp40: ml1488, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1481, ml1488)

    # U34: ml1481, upper_triangular, tmp41: ml1488, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1481, ml1488)

    # tmp17: ml1488, full
    # x = tmp17
    return (ml1488)
end