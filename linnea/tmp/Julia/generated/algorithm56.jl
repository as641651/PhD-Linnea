using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm56(ml1872::Array{Float64,2}, ml1873::Array{Float64,2}, ml1874::Array{Float64,2}, ml1875::Array{Float64,2}, ml1876::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1872, full, L: ml1873, full, A: ml1874, full, B: ml1875, full, y: ml1876, full
    ml1877 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1874, ml1877, info) = LinearAlgebra.LAPACK.getrf!(ml1874)

    # R: ml1872, full, L: ml1873, full, B: ml1875, full, y: ml1876, full, P11: ml1877, ipiv, L9: ml1874, lower_triangular_udiag, U10: ml1874, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1874, ml1875)

    # R: ml1872, full, L: ml1873, full, y: ml1876, full, P11: ml1877, ipiv, L9: ml1874, lower_triangular_udiag, tmp53: ml1875, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1874, ml1875)

    # R: ml1872, full, L: ml1873, full, y: ml1876, full, P11: ml1877, ipiv, tmp54: ml1875, full
    ml1878 = [1:length(ml1877);]
    @inbounds for i in 1:length(ml1877)
        ml1878[i], ml1878[ml1877[i]] = ml1878[ml1877[i]], ml1878[i];
    end;
    ml1879 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1879 = ml1875[:,invperm(ml1878)]

    # R: ml1872, full, L: ml1873, full, y: ml1876, full, tmp55: ml1879, full
    ml1880 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1880, ml1879)

    # R: ml1872, full, L: ml1873, full, y: ml1876, full, tmp25: ml1880, full
    ml1881 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1880, 0.0, ml1881)

    # R: ml1872, full, L: ml1873, full, y: ml1876, full, tmp19: ml1881, symmetric_lower_triangular
    ml1882 = diag(ml1873)
    ml1883 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1872, 1, ml1883, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1872, 2);
        view(ml1872, :, i)[:] .*= ml1882;
    end;        

    # R: ml1883, full, y: ml1876, full, tmp19: ml1881, symmetric_lower_triangular, tmp29: ml1872, full
    for i = 1:2000-1;
        view(ml1881, i, i+1:2000)[:] = view(ml1881, i+1:2000, i);
    end;
    ml1884 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1881, 1, ml1884, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1872, ml1883, 1.0, ml1881)

    # y: ml1876, full, tmp19: ml1884, full, tmp31: ml1881, full
    ml1885 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1881, ml1885, info) = LinearAlgebra.LAPACK.getrf!(ml1881)

    # y: ml1876, full, tmp19: ml1884, full, P35: ml1885, ipiv, L33: ml1881, lower_triangular_udiag, U34: ml1881, upper_triangular
    ml1886 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1884, ml1876, 0.0, ml1886)

    # P35: ml1885, ipiv, L33: ml1881, lower_triangular_udiag, U34: ml1881, upper_triangular, tmp32: ml1886, full
    ml1887 = [1:length(ml1885);]
    @inbounds for i in 1:length(ml1885)
        ml1887[i], ml1887[ml1885[i]] = ml1887[ml1885[i]], ml1887[i];
    end;
    ml1888 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1888 = ml1886[ml1887]

    # L33: ml1881, lower_triangular_udiag, U34: ml1881, upper_triangular, tmp40: ml1888, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1881, ml1888)

    # U34: ml1881, upper_triangular, tmp41: ml1888, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1881, ml1888)

    # tmp17: ml1888, full
    # x = tmp17
    return (ml1888)
end