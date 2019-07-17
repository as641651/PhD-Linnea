using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm58(ml1940::Array{Float64,2}, ml1941::Array{Float64,2}, ml1942::Array{Float64,2}, ml1943::Array{Float64,2}, ml1944::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1940, full, L: ml1941, full, A: ml1942, full, B: ml1943, full, y: ml1944, full
    ml1945 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1942, ml1945, info) = LinearAlgebra.LAPACK.getrf!(ml1942)

    # R: ml1940, full, L: ml1941, full, B: ml1943, full, y: ml1944, full, P11: ml1945, ipiv, L9: ml1942, lower_triangular_udiag, U10: ml1942, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1942, ml1943)

    # R: ml1940, full, L: ml1941, full, y: ml1944, full, P11: ml1945, ipiv, L9: ml1942, lower_triangular_udiag, tmp53: ml1943, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1942, ml1943)

    # R: ml1940, full, L: ml1941, full, y: ml1944, full, P11: ml1945, ipiv, tmp54: ml1943, full
    ml1946 = [1:length(ml1945);]
    @inbounds for i in 1:length(ml1945)
        ml1946[i], ml1946[ml1945[i]] = ml1946[ml1945[i]], ml1946[i];
    end;
    ml1947 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1947 = ml1943[:,invperm(ml1946)]

    # R: ml1940, full, L: ml1941, full, y: ml1944, full, tmp55: ml1947, full
    ml1948 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1948, ml1947)

    # R: ml1940, full, L: ml1941, full, y: ml1944, full, tmp25: ml1948, full
    ml1949 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1948, 0.0, ml1949)

    # R: ml1940, full, L: ml1941, full, y: ml1944, full, tmp19: ml1949, symmetric_lower_triangular
    ml1950 = diag(ml1941)
    ml1951 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1940, 1, ml1951, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1940, 2);
        view(ml1940, :, i)[:] .*= ml1950;
    end;        

    # R: ml1951, full, y: ml1944, full, tmp19: ml1949, symmetric_lower_triangular, tmp29: ml1940, full
    for i = 1:2000-1;
        view(ml1949, i, i+1:2000)[:] = view(ml1949, i+1:2000, i);
    end;
    ml1952 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1949, 1, ml1952, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1940, ml1951, 1.0, ml1949)

    # y: ml1944, full, tmp19: ml1952, full, tmp31: ml1949, full
    ml1953 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1949, ml1953, info) = LinearAlgebra.LAPACK.getrf!(ml1949)

    # y: ml1944, full, tmp19: ml1952, full, P35: ml1953, ipiv, L33: ml1949, lower_triangular_udiag, U34: ml1949, upper_triangular
    ml1954 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1952, ml1944, 0.0, ml1954)

    # P35: ml1953, ipiv, L33: ml1949, lower_triangular_udiag, U34: ml1949, upper_triangular, tmp32: ml1954, full
    ml1955 = [1:length(ml1953);]
    @inbounds for i in 1:length(ml1953)
        ml1955[i], ml1955[ml1953[i]] = ml1955[ml1953[i]], ml1955[i];
    end;
    ml1956 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1956 = ml1954[ml1955]

    # L33: ml1949, lower_triangular_udiag, U34: ml1949, upper_triangular, tmp40: ml1956, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1949, ml1956)

    # U34: ml1949, upper_triangular, tmp41: ml1956, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1949, ml1956)

    # tmp17: ml1956, full
    # x = tmp17
    return (ml1956)
end