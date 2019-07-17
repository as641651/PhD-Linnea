using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm53(ml1772::Array{Float64,2}, ml1773::Array{Float64,2}, ml1774::Array{Float64,2}, ml1775::Array{Float64,2}, ml1776::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1772, full, L: ml1773, full, A: ml1774, full, B: ml1775, full, y: ml1776, full
    ml1777 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1774, ml1777, info) = LinearAlgebra.LAPACK.getrf!(ml1774)

    # R: ml1772, full, L: ml1773, full, B: ml1775, full, y: ml1776, full, P11: ml1777, ipiv, L9: ml1774, lower_triangular_udiag, U10: ml1774, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1774, ml1775)

    # R: ml1772, full, L: ml1773, full, y: ml1776, full, P11: ml1777, ipiv, L9: ml1774, lower_triangular_udiag, tmp53: ml1775, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1774, ml1775)

    # R: ml1772, full, L: ml1773, full, y: ml1776, full, P11: ml1777, ipiv, tmp54: ml1775, full
    ml1778 = [1:length(ml1777);]
    @inbounds for i in 1:length(ml1777)
        ml1778[i], ml1778[ml1777[i]] = ml1778[ml1777[i]], ml1778[i];
    end;
    ml1779 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1779 = ml1775[:,invperm(ml1778)]

    # R: ml1772, full, L: ml1773, full, y: ml1776, full, tmp55: ml1779, full
    ml1780 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1780, ml1779)

    # R: ml1772, full, L: ml1773, full, y: ml1776, full, tmp25: ml1780, full
    ml1781 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1780, 0.0, ml1781)

    # R: ml1772, full, L: ml1773, full, y: ml1776, full, tmp19: ml1781, symmetric_lower_triangular
    ml1782 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1781, ml1776, 0.0, ml1782)

    # R: ml1772, full, L: ml1773, full, tmp19: ml1781, symmetric_lower_triangular, tmp32: ml1782, full
    ml1783 = diag(ml1773)
    ml1784 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1772, 1, ml1784, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1772, 2);
        view(ml1772, :, i)[:] .*= ml1783;
    end;        

    # R: ml1784, full, tmp19: ml1781, symmetric_lower_triangular, tmp32: ml1782, full, tmp29: ml1772, full
    for i = 1:2000-1;
        view(ml1781, i, i+1:2000)[:] = view(ml1781, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1772, ml1784, 1.0, ml1781)

    # tmp32: ml1782, full, tmp31: ml1781, full
    ml1785 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1781, ml1785, info) = LinearAlgebra.LAPACK.getrf!(ml1781)

    # tmp32: ml1782, full, P35: ml1785, ipiv, L33: ml1781, lower_triangular_udiag, U34: ml1781, upper_triangular
    ml1786 = [1:length(ml1785);]
    @inbounds for i in 1:length(ml1785)
        ml1786[i], ml1786[ml1785[i]] = ml1786[ml1785[i]], ml1786[i];
    end;
    ml1787 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1787 = ml1782[ml1786]

    # L33: ml1781, lower_triangular_udiag, U34: ml1781, upper_triangular, tmp40: ml1787, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1781, ml1787)

    # U34: ml1781, upper_triangular, tmp41: ml1787, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1781, ml1787)

    # tmp17: ml1787, full
    # x = tmp17
    return (ml1787)
end