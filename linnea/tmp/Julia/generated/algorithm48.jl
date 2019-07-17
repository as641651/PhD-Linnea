using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm48(ml1608::Array{Float64,2}, ml1609::Array{Float64,2}, ml1610::Array{Float64,2}, ml1611::Array{Float64,2}, ml1612::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1608, full, L: ml1609, full, A: ml1610, full, B: ml1611, full, y: ml1612, full
    ml1613 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1613, ml1611)

    # R: ml1608, full, L: ml1609, full, A: ml1610, full, y: ml1612, full, tmp26: ml1613, full
    ml1614 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1610, ml1614, info) = LinearAlgebra.LAPACK.getrf!(ml1610)

    # R: ml1608, full, L: ml1609, full, y: ml1612, full, tmp26: ml1613, full, P11: ml1614, ipiv, L9: ml1610, lower_triangular_udiag, U10: ml1610, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1610, ml1613)

    # R: ml1608, full, L: ml1609, full, y: ml1612, full, P11: ml1614, ipiv, L9: ml1610, lower_triangular_udiag, tmp27: ml1613, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1610, ml1613)

    # R: ml1608, full, L: ml1609, full, y: ml1612, full, P11: ml1614, ipiv, tmp28: ml1613, full
    ml1615 = [1:length(ml1614);]
    @inbounds for i in 1:length(ml1614)
        ml1615[i], ml1615[ml1614[i]] = ml1615[ml1614[i]], ml1615[i];
    end;
    ml1616 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1616 = ml1613[invperm(ml1615),:]

    # R: ml1608, full, L: ml1609, full, y: ml1612, full, tmp25: ml1616, full
    ml1617 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1616, 0.0, ml1617)

    # R: ml1608, full, L: ml1609, full, y: ml1612, full, tmp19: ml1617, symmetric_lower_triangular
    ml1618 = diag(ml1609)
    ml1619 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1608, 1, ml1619, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1608, 2);
        view(ml1608, :, i)[:] .*= ml1618;
    end;        

    # R: ml1619, full, y: ml1612, full, tmp19: ml1617, symmetric_lower_triangular, tmp29: ml1608, full
    for i = 1:2000-1;
        view(ml1617, i, i+1:2000)[:] = view(ml1617, i+1:2000, i);
    end;
    ml1620 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1617, 1, ml1620, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1608, ml1619, 1.0, ml1617)

    # y: ml1612, full, tmp19: ml1620, full, tmp31: ml1617, full
    ml1621 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1617, ml1621, info) = LinearAlgebra.LAPACK.getrf!(ml1617)

    # y: ml1612, full, tmp19: ml1620, full, P35: ml1621, ipiv, L33: ml1617, lower_triangular_udiag, U34: ml1617, upper_triangular
    ml1622 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1620, ml1612, 0.0, ml1622)

    # P35: ml1621, ipiv, L33: ml1617, lower_triangular_udiag, U34: ml1617, upper_triangular, tmp32: ml1622, full
    ml1623 = [1:length(ml1621);]
    @inbounds for i in 1:length(ml1621)
        ml1623[i], ml1623[ml1621[i]] = ml1623[ml1621[i]], ml1623[i];
    end;
    ml1624 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1624 = ml1622[ml1623]

    # L33: ml1617, lower_triangular_udiag, U34: ml1617, upper_triangular, tmp40: ml1624, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1617, ml1624)

    # U34: ml1617, upper_triangular, tmp41: ml1624, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1617, ml1624)

    # tmp17: ml1624, full
    # x = tmp17
    return (ml1624)
end