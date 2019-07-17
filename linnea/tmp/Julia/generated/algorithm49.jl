using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm49(ml1642::Array{Float64,2}, ml1643::Array{Float64,2}, ml1644::Array{Float64,2}, ml1645::Array{Float64,2}, ml1646::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1642, full, L: ml1643, full, A: ml1644, full, B: ml1645, full, y: ml1646, full
    ml1647 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1647, ml1645)

    # R: ml1642, full, L: ml1643, full, A: ml1644, full, y: ml1646, full, tmp26: ml1647, full
    ml1648 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1644, ml1648, info) = LinearAlgebra.LAPACK.getrf!(ml1644)

    # R: ml1642, full, L: ml1643, full, y: ml1646, full, tmp26: ml1647, full, P11: ml1648, ipiv, L9: ml1644, lower_triangular_udiag, U10: ml1644, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1644, ml1647)

    # R: ml1642, full, L: ml1643, full, y: ml1646, full, P11: ml1648, ipiv, L9: ml1644, lower_triangular_udiag, tmp27: ml1647, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1644, ml1647)

    # R: ml1642, full, L: ml1643, full, y: ml1646, full, P11: ml1648, ipiv, tmp28: ml1647, full
    ml1649 = [1:length(ml1648);]
    @inbounds for i in 1:length(ml1648)
        ml1649[i], ml1649[ml1648[i]] = ml1649[ml1648[i]], ml1649[i];
    end;
    ml1650 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1650 = ml1647[invperm(ml1649),:]

    # R: ml1642, full, L: ml1643, full, y: ml1646, full, tmp25: ml1650, full
    ml1651 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1650, 0.0, ml1651)

    # R: ml1642, full, L: ml1643, full, y: ml1646, full, tmp19: ml1651, symmetric_lower_triangular
    ml1652 = diag(ml1643)
    ml1653 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1642, 1, ml1653, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1642, 2);
        view(ml1642, :, i)[:] .*= ml1652;
    end;        

    # R: ml1653, full, y: ml1646, full, tmp19: ml1651, symmetric_lower_triangular, tmp29: ml1642, full
    for i = 1:2000-1;
        view(ml1651, i, i+1:2000)[:] = view(ml1651, i+1:2000, i);
    end;
    ml1654 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1651, 1, ml1654, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1642, ml1653, 1.0, ml1651)

    # y: ml1646, full, tmp19: ml1654, full, tmp31: ml1651, full
    ml1655 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1651, ml1655, info) = LinearAlgebra.LAPACK.getrf!(ml1651)

    # y: ml1646, full, tmp19: ml1654, full, P35: ml1655, ipiv, L33: ml1651, lower_triangular_udiag, U34: ml1651, upper_triangular
    ml1656 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1654, ml1646, 0.0, ml1656)

    # P35: ml1655, ipiv, L33: ml1651, lower_triangular_udiag, U34: ml1651, upper_triangular, tmp32: ml1656, full
    ml1657 = [1:length(ml1655);]
    @inbounds for i in 1:length(ml1655)
        ml1657[i], ml1657[ml1655[i]] = ml1657[ml1655[i]], ml1657[i];
    end;
    ml1658 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1658 = ml1656[ml1657]

    # L33: ml1651, lower_triangular_udiag, U34: ml1651, upper_triangular, tmp40: ml1658, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1651, ml1658)

    # U34: ml1651, upper_triangular, tmp41: ml1658, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1651, ml1658)

    # tmp17: ml1658, full
    # x = tmp17
    return (ml1658)
end