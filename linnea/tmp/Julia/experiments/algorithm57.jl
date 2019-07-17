using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm57(ml1923::Array{Float64,2}, ml1924::Array{Float64,2}, ml1925::Array{Float64,2}, ml1926::Array{Float64,2}, ml1927::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1923, full, L: ml1924, full, A: ml1925, full, B: ml1926, full, y: ml1927, full
    ml1928 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1925, ml1928, info) = LinearAlgebra.LAPACK.getrf!(ml1925)

    # R: ml1923, full, L: ml1924, full, B: ml1926, full, y: ml1927, full, P11: ml1928, ipiv, L9: ml1925, lower_triangular_udiag, U10: ml1925, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1925, ml1926)

    # R: ml1923, full, L: ml1924, full, y: ml1927, full, P11: ml1928, ipiv, L9: ml1925, lower_triangular_udiag, tmp53: ml1926, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1925, ml1926)

    # R: ml1923, full, L: ml1924, full, y: ml1927, full, P11: ml1928, ipiv, tmp54: ml1926, full
    ml1929 = [1:length(ml1928);]
    @inbounds for i in 1:length(ml1928)
        ml1929[i], ml1929[ml1928[i]] = ml1929[ml1928[i]], ml1929[i];
    end;
    ml1930 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1930 = ml1926[:,invperm(ml1929)]

    # R: ml1923, full, L: ml1924, full, y: ml1927, full, tmp55: ml1930, full
    ml1931 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1931, ml1930)

    # R: ml1923, full, L: ml1924, full, y: ml1927, full, tmp25: ml1931, full
    ml1932 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1931, 0.0, ml1932)

    # R: ml1923, full, L: ml1924, full, y: ml1927, full, tmp19: ml1932, symmetric_lower_triangular
    ml1933 = diag(ml1924)
    ml1934 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1923, 1, ml1934, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1923, 2);
        view(ml1923, :, i)[:] .*= ml1933;
    end;        

    # R: ml1934, full, y: ml1927, full, tmp19: ml1932, symmetric_lower_triangular, tmp29: ml1923, full
    for i = 1:2000-1;
        view(ml1932, i, i+1:2000)[:] = view(ml1932, i+1:2000, i);
    end;
    ml1935 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1932, 1, ml1935, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1923, ml1934, 1.0, ml1932)

    # y: ml1927, full, tmp19: ml1935, full, tmp31: ml1932, full
    ml1936 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1932, ml1936, info) = LinearAlgebra.LAPACK.getrf!(ml1932)

    # y: ml1927, full, tmp19: ml1935, full, P35: ml1936, ipiv, L33: ml1932, lower_triangular_udiag, U34: ml1932, upper_triangular
    ml1937 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1935, ml1927, 0.0, ml1937)

    # P35: ml1936, ipiv, L33: ml1932, lower_triangular_udiag, U34: ml1932, upper_triangular, tmp32: ml1937, full
    ml1938 = [1:length(ml1936);]
    @inbounds for i in 1:length(ml1936)
        ml1938[i], ml1938[ml1936[i]] = ml1938[ml1936[i]], ml1938[i];
    end;
    ml1939 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1939 = ml1937[ml1938]

    # L33: ml1932, lower_triangular_udiag, U34: ml1932, upper_triangular, tmp40: ml1939, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1932, ml1939)

    # U34: ml1932, upper_triangular, tmp41: ml1939, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1932, ml1939)

    # tmp17: ml1939, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1939), (finish-start)*1e-9)
end