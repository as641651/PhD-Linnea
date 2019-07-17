using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm54(ml1804::Array{Float64,2}, ml1805::Array{Float64,2}, ml1806::Array{Float64,2}, ml1807::Array{Float64,2}, ml1808::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1804, full, L: ml1805, full, A: ml1806, full, B: ml1807, full, y: ml1808, full
    ml1809 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1809, ml1807)

    # R: ml1804, full, L: ml1805, full, A: ml1806, full, y: ml1808, full, tmp26: ml1809, full
    ml1810 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1806, ml1810, info) = LinearAlgebra.LAPACK.getrf!(ml1806)

    # R: ml1804, full, L: ml1805, full, y: ml1808, full, tmp26: ml1809, full, P11: ml1810, ipiv, L9: ml1806, lower_triangular_udiag, U10: ml1806, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1806, ml1809)

    # R: ml1804, full, L: ml1805, full, y: ml1808, full, P11: ml1810, ipiv, L9: ml1806, lower_triangular_udiag, tmp27: ml1809, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1806, ml1809)

    # R: ml1804, full, L: ml1805, full, y: ml1808, full, P11: ml1810, ipiv, tmp28: ml1809, full
    ml1811 = [1:length(ml1810);]
    @inbounds for i in 1:length(ml1810)
        ml1811[i], ml1811[ml1810[i]] = ml1811[ml1810[i]], ml1811[i];
    end;
    ml1812 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1812 = ml1809[invperm(ml1811),:]

    # R: ml1804, full, L: ml1805, full, y: ml1808, full, tmp25: ml1812, full
    ml1813 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1812, 0.0, ml1813)

    # R: ml1804, full, L: ml1805, full, y: ml1808, full, tmp19: ml1813, symmetric_lower_triangular
    ml1814 = diag(ml1805)
    ml1815 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1804, 1, ml1815, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1804, 2);
        view(ml1804, :, i)[:] .*= ml1814;
    end;        

    # R: ml1815, full, y: ml1808, full, tmp19: ml1813, symmetric_lower_triangular, tmp29: ml1804, full
    for i = 1:2000-1;
        view(ml1813, i, i+1:2000)[:] = view(ml1813, i+1:2000, i);
    end;
    ml1816 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1813, 1, ml1816, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1804, ml1815, 1.0, ml1813)

    # y: ml1808, full, tmp19: ml1816, full, tmp31: ml1813, full
    ml1817 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1813, ml1817, info) = LinearAlgebra.LAPACK.getrf!(ml1813)

    # y: ml1808, full, tmp19: ml1816, full, P35: ml1817, ipiv, L33: ml1813, lower_triangular_udiag, U34: ml1813, upper_triangular
    ml1818 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1816, ml1808, 0.0, ml1818)

    # P35: ml1817, ipiv, L33: ml1813, lower_triangular_udiag, U34: ml1813, upper_triangular, tmp32: ml1818, full
    ml1819 = [1:length(ml1817);]
    @inbounds for i in 1:length(ml1817)
        ml1819[i], ml1819[ml1817[i]] = ml1819[ml1817[i]], ml1819[i];
    end;
    ml1820 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1820 = ml1818[ml1819]

    # L33: ml1813, lower_triangular_udiag, U34: ml1813, upper_triangular, tmp40: ml1820, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1813, ml1820)

    # U34: ml1813, upper_triangular, tmp41: ml1820, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1813, ml1820)

    # tmp17: ml1820, full
    # x = tmp17
    return (ml1820)
end