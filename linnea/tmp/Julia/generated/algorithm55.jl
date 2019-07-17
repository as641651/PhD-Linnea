using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm55(ml1838::Array{Float64,2}, ml1839::Array{Float64,2}, ml1840::Array{Float64,2}, ml1841::Array{Float64,2}, ml1842::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1838, full, L: ml1839, full, A: ml1840, full, B: ml1841, full, y: ml1842, full
    ml1843 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1840, ml1843, info) = LinearAlgebra.LAPACK.getrf!(ml1840)

    # R: ml1838, full, L: ml1839, full, B: ml1841, full, y: ml1842, full, P11: ml1843, ipiv, L9: ml1840, lower_triangular_udiag, U10: ml1840, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1840, ml1841)

    # R: ml1838, full, L: ml1839, full, y: ml1842, full, P11: ml1843, ipiv, L9: ml1840, lower_triangular_udiag, tmp53: ml1841, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1840, ml1841)

    # R: ml1838, full, L: ml1839, full, y: ml1842, full, P11: ml1843, ipiv, tmp54: ml1841, full
    ml1844 = [1:length(ml1843);]
    @inbounds for i in 1:length(ml1843)
        ml1844[i], ml1844[ml1843[i]] = ml1844[ml1843[i]], ml1844[i];
    end;
    ml1845 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1845 = ml1841[:,invperm(ml1844)]

    # R: ml1838, full, L: ml1839, full, y: ml1842, full, tmp55: ml1845, full
    ml1846 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1846, ml1845)

    # R: ml1838, full, L: ml1839, full, y: ml1842, full, tmp25: ml1846, full
    ml1847 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1846, 0.0, ml1847)

    # R: ml1838, full, L: ml1839, full, y: ml1842, full, tmp19: ml1847, symmetric_lower_triangular
    ml1848 = diag(ml1839)
    ml1849 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1838, 1, ml1849, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1838, 2);
        view(ml1838, :, i)[:] .*= ml1848;
    end;        

    # R: ml1849, full, y: ml1842, full, tmp19: ml1847, symmetric_lower_triangular, tmp29: ml1838, full
    for i = 1:2000-1;
        view(ml1847, i, i+1:2000)[:] = view(ml1847, i+1:2000, i);
    end;
    ml1850 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1847, 1, ml1850, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1838, ml1849, 1.0, ml1847)

    # y: ml1842, full, tmp19: ml1850, full, tmp31: ml1847, full
    ml1851 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1847, ml1851, info) = LinearAlgebra.LAPACK.getrf!(ml1847)

    # y: ml1842, full, tmp19: ml1850, full, P35: ml1851, ipiv, L33: ml1847, lower_triangular_udiag, U34: ml1847, upper_triangular
    ml1852 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1850, ml1842, 0.0, ml1852)

    # P35: ml1851, ipiv, L33: ml1847, lower_triangular_udiag, U34: ml1847, upper_triangular, tmp32: ml1852, full
    ml1853 = [1:length(ml1851);]
    @inbounds for i in 1:length(ml1851)
        ml1853[i], ml1853[ml1851[i]] = ml1853[ml1851[i]], ml1853[i];
    end;
    ml1854 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1854 = ml1852[ml1853]

    # L33: ml1847, lower_triangular_udiag, U34: ml1847, upper_triangular, tmp40: ml1854, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1847, ml1854)

    # U34: ml1847, upper_triangular, tmp41: ml1854, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1847, ml1854)

    # tmp17: ml1854, full
    # x = tmp17
    return (ml1854)
end