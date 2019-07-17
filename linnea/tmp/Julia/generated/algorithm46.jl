using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm46(ml1540::Array{Float64,2}, ml1541::Array{Float64,2}, ml1542::Array{Float64,2}, ml1543::Array{Float64,2}, ml1544::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1540, full, L: ml1541, full, A: ml1542, full, B: ml1543, full, y: ml1544, full
    ml1545 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1545, ml1543)

    # R: ml1540, full, L: ml1541, full, A: ml1542, full, y: ml1544, full, tmp26: ml1545, full
    ml1546 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1542, ml1546, info) = LinearAlgebra.LAPACK.getrf!(ml1542)

    # R: ml1540, full, L: ml1541, full, y: ml1544, full, tmp26: ml1545, full, P11: ml1546, ipiv, L9: ml1542, lower_triangular_udiag, U10: ml1542, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1542, ml1545)

    # R: ml1540, full, L: ml1541, full, y: ml1544, full, P11: ml1546, ipiv, L9: ml1542, lower_triangular_udiag, tmp27: ml1545, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1542, ml1545)

    # R: ml1540, full, L: ml1541, full, y: ml1544, full, P11: ml1546, ipiv, tmp28: ml1545, full
    ml1547 = [1:length(ml1546);]
    @inbounds for i in 1:length(ml1546)
        ml1547[i], ml1547[ml1546[i]] = ml1547[ml1546[i]], ml1547[i];
    end;
    ml1548 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1548 = ml1545[invperm(ml1547),:]

    # R: ml1540, full, L: ml1541, full, y: ml1544, full, tmp25: ml1548, full
    ml1549 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1548, 0.0, ml1549)

    # R: ml1540, full, L: ml1541, full, y: ml1544, full, tmp19: ml1549, symmetric_lower_triangular
    ml1550 = diag(ml1541)
    ml1551 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1540, 1, ml1551, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1540, 2);
        view(ml1540, :, i)[:] .*= ml1550;
    end;        

    # R: ml1551, full, y: ml1544, full, tmp19: ml1549, symmetric_lower_triangular, tmp29: ml1540, full
    for i = 1:2000-1;
        view(ml1549, i, i+1:2000)[:] = view(ml1549, i+1:2000, i);
    end;
    ml1552 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1549, 1, ml1552, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1540, ml1551, 1.0, ml1549)

    # y: ml1544, full, tmp19: ml1552, full, tmp31: ml1549, full
    ml1553 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1549, ml1553, info) = LinearAlgebra.LAPACK.getrf!(ml1549)

    # y: ml1544, full, tmp19: ml1552, full, P35: ml1553, ipiv, L33: ml1549, lower_triangular_udiag, U34: ml1549, upper_triangular
    ml1554 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1552, ml1544, 0.0, ml1554)

    # P35: ml1553, ipiv, L33: ml1549, lower_triangular_udiag, U34: ml1549, upper_triangular, tmp32: ml1554, full
    ml1555 = [1:length(ml1553);]
    @inbounds for i in 1:length(ml1553)
        ml1555[i], ml1555[ml1553[i]] = ml1555[ml1553[i]], ml1555[i];
    end;
    ml1556 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1556 = ml1554[ml1555]

    # L33: ml1549, lower_triangular_udiag, U34: ml1549, upper_triangular, tmp40: ml1556, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1549, ml1556)

    # U34: ml1549, upper_triangular, tmp41: ml1556, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1549, ml1556)

    # tmp17: ml1556, full
    # x = tmp17
    return (ml1556)
end