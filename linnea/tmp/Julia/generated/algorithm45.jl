using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm45(ml1506::Array{Float64,2}, ml1507::Array{Float64,2}, ml1508::Array{Float64,2}, ml1509::Array{Float64,2}, ml1510::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1506, full, L: ml1507, full, A: ml1508, full, B: ml1509, full, y: ml1510, full
    ml1511 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1511, ml1509)

    # R: ml1506, full, L: ml1507, full, A: ml1508, full, y: ml1510, full, tmp26: ml1511, full
    ml1512 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1508, ml1512, info) = LinearAlgebra.LAPACK.getrf!(ml1508)

    # R: ml1506, full, L: ml1507, full, y: ml1510, full, tmp26: ml1511, full, P11: ml1512, ipiv, L9: ml1508, lower_triangular_udiag, U10: ml1508, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1508, ml1511)

    # R: ml1506, full, L: ml1507, full, y: ml1510, full, P11: ml1512, ipiv, L9: ml1508, lower_triangular_udiag, tmp27: ml1511, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1508, ml1511)

    # R: ml1506, full, L: ml1507, full, y: ml1510, full, P11: ml1512, ipiv, tmp28: ml1511, full
    ml1513 = [1:length(ml1512);]
    @inbounds for i in 1:length(ml1512)
        ml1513[i], ml1513[ml1512[i]] = ml1513[ml1512[i]], ml1513[i];
    end;
    ml1514 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1514 = ml1511[invperm(ml1513),:]

    # R: ml1506, full, L: ml1507, full, y: ml1510, full, tmp25: ml1514, full
    ml1515 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1514, 0.0, ml1515)

    # R: ml1506, full, L: ml1507, full, y: ml1510, full, tmp19: ml1515, symmetric_lower_triangular
    ml1516 = diag(ml1507)
    ml1517 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1506, 1, ml1517, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1506, 2);
        view(ml1506, :, i)[:] .*= ml1516;
    end;        

    # R: ml1517, full, y: ml1510, full, tmp19: ml1515, symmetric_lower_triangular, tmp29: ml1506, full
    for i = 1:2000-1;
        view(ml1515, i, i+1:2000)[:] = view(ml1515, i+1:2000, i);
    end;
    ml1518 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1515, 1, ml1518, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1506, ml1517, 1.0, ml1515)

    # y: ml1510, full, tmp19: ml1518, full, tmp31: ml1515, full
    ml1519 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1515, ml1519, info) = LinearAlgebra.LAPACK.getrf!(ml1515)

    # y: ml1510, full, tmp19: ml1518, full, P35: ml1519, ipiv, L33: ml1515, lower_triangular_udiag, U34: ml1515, upper_triangular
    ml1520 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1518, ml1510, 0.0, ml1520)

    # P35: ml1519, ipiv, L33: ml1515, lower_triangular_udiag, U34: ml1515, upper_triangular, tmp32: ml1520, full
    ml1521 = [1:length(ml1519);]
    @inbounds for i in 1:length(ml1519)
        ml1521[i], ml1521[ml1519[i]] = ml1521[ml1519[i]], ml1521[i];
    end;
    ml1522 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1522 = ml1520[ml1521]

    # L33: ml1515, lower_triangular_udiag, U34: ml1515, upper_triangular, tmp40: ml1522, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1515, ml1522)

    # U34: ml1515, upper_triangular, tmp41: ml1522, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1515, ml1522)

    # tmp17: ml1522, full
    # x = tmp17
    return (ml1522)
end