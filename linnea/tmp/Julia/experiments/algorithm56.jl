using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm56(ml1889::Array{Float64,2}, ml1890::Array{Float64,2}, ml1891::Array{Float64,2}, ml1892::Array{Float64,2}, ml1893::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1889, full, L: ml1890, full, A: ml1891, full, B: ml1892, full, y: ml1893, full
    ml1894 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1891, ml1894, info) = LinearAlgebra.LAPACK.getrf!(ml1891)

    # R: ml1889, full, L: ml1890, full, B: ml1892, full, y: ml1893, full, P11: ml1894, ipiv, L9: ml1891, lower_triangular_udiag, U10: ml1891, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1891, ml1892)

    # R: ml1889, full, L: ml1890, full, y: ml1893, full, P11: ml1894, ipiv, L9: ml1891, lower_triangular_udiag, tmp53: ml1892, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1891, ml1892)

    # R: ml1889, full, L: ml1890, full, y: ml1893, full, P11: ml1894, ipiv, tmp54: ml1892, full
    ml1895 = [1:length(ml1894);]
    @inbounds for i in 1:length(ml1894)
        ml1895[i], ml1895[ml1894[i]] = ml1895[ml1894[i]], ml1895[i];
    end;
    ml1896 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1896 = ml1892[:,invperm(ml1895)]

    # R: ml1889, full, L: ml1890, full, y: ml1893, full, tmp55: ml1896, full
    ml1897 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1897, ml1896)

    # R: ml1889, full, L: ml1890, full, y: ml1893, full, tmp25: ml1897, full
    ml1898 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1897, 0.0, ml1898)

    # R: ml1889, full, L: ml1890, full, y: ml1893, full, tmp19: ml1898, symmetric_lower_triangular
    ml1899 = diag(ml1890)
    ml1900 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1889, 1, ml1900, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1889, 2);
        view(ml1889, :, i)[:] .*= ml1899;
    end;        

    # R: ml1900, full, y: ml1893, full, tmp19: ml1898, symmetric_lower_triangular, tmp29: ml1889, full
    for i = 1:2000-1;
        view(ml1898, i, i+1:2000)[:] = view(ml1898, i+1:2000, i);
    end;
    ml1901 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1898, 1, ml1901, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1889, ml1900, 1.0, ml1898)

    # y: ml1893, full, tmp19: ml1901, full, tmp31: ml1898, full
    ml1902 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1898, ml1902, info) = LinearAlgebra.LAPACK.getrf!(ml1898)

    # y: ml1893, full, tmp19: ml1901, full, P35: ml1902, ipiv, L33: ml1898, lower_triangular_udiag, U34: ml1898, upper_triangular
    ml1903 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1901, ml1893, 0.0, ml1903)

    # P35: ml1902, ipiv, L33: ml1898, lower_triangular_udiag, U34: ml1898, upper_triangular, tmp32: ml1903, full
    ml1904 = [1:length(ml1902);]
    @inbounds for i in 1:length(ml1902)
        ml1904[i], ml1904[ml1902[i]] = ml1904[ml1902[i]], ml1904[i];
    end;
    ml1905 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1905 = ml1903[ml1904]

    # L33: ml1898, lower_triangular_udiag, U34: ml1898, upper_triangular, tmp40: ml1905, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1898, ml1905)

    # U34: ml1898, upper_triangular, tmp41: ml1905, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1898, ml1905)

    # tmp17: ml1905, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1905), (finish-start)*1e-9)
end