using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm58(ml1957::Array{Float64,2}, ml1958::Array{Float64,2}, ml1959::Array{Float64,2}, ml1960::Array{Float64,2}, ml1961::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1957, full, L: ml1958, full, A: ml1959, full, B: ml1960, full, y: ml1961, full
    ml1962 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1959, ml1962, info) = LinearAlgebra.LAPACK.getrf!(ml1959)

    # R: ml1957, full, L: ml1958, full, B: ml1960, full, y: ml1961, full, P11: ml1962, ipiv, L9: ml1959, lower_triangular_udiag, U10: ml1959, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1959, ml1960)

    # R: ml1957, full, L: ml1958, full, y: ml1961, full, P11: ml1962, ipiv, L9: ml1959, lower_triangular_udiag, tmp53: ml1960, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1959, ml1960)

    # R: ml1957, full, L: ml1958, full, y: ml1961, full, P11: ml1962, ipiv, tmp54: ml1960, full
    ml1963 = [1:length(ml1962);]
    @inbounds for i in 1:length(ml1962)
        ml1963[i], ml1963[ml1962[i]] = ml1963[ml1962[i]], ml1963[i];
    end;
    ml1964 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1964 = ml1960[:,invperm(ml1963)]

    # R: ml1957, full, L: ml1958, full, y: ml1961, full, tmp55: ml1964, full
    ml1965 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1965, ml1964)

    # R: ml1957, full, L: ml1958, full, y: ml1961, full, tmp25: ml1965, full
    ml1966 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1965, 0.0, ml1966)

    # R: ml1957, full, L: ml1958, full, y: ml1961, full, tmp19: ml1966, symmetric_lower_triangular
    ml1967 = diag(ml1958)
    ml1968 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1957, 1, ml1968, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1957, 2);
        view(ml1957, :, i)[:] .*= ml1967;
    end;        

    # R: ml1968, full, y: ml1961, full, tmp19: ml1966, symmetric_lower_triangular, tmp29: ml1957, full
    for i = 1:2000-1;
        view(ml1966, i, i+1:2000)[:] = view(ml1966, i+1:2000, i);
    end;
    ml1969 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1966, 1, ml1969, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1957, ml1968, 1.0, ml1966)

    # y: ml1961, full, tmp19: ml1969, full, tmp31: ml1966, full
    ml1970 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1966, ml1970, info) = LinearAlgebra.LAPACK.getrf!(ml1966)

    # y: ml1961, full, tmp19: ml1969, full, P35: ml1970, ipiv, L33: ml1966, lower_triangular_udiag, U34: ml1966, upper_triangular
    ml1971 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1969, ml1961, 0.0, ml1971)

    # P35: ml1970, ipiv, L33: ml1966, lower_triangular_udiag, U34: ml1966, upper_triangular, tmp32: ml1971, full
    ml1972 = [1:length(ml1970);]
    @inbounds for i in 1:length(ml1970)
        ml1972[i], ml1972[ml1970[i]] = ml1972[ml1970[i]], ml1972[i];
    end;
    ml1973 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1973 = ml1971[ml1972]

    # L33: ml1966, lower_triangular_udiag, U34: ml1966, upper_triangular, tmp40: ml1973, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1966, ml1973)

    # U34: ml1966, upper_triangular, tmp41: ml1973, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1966, ml1973)

    # tmp17: ml1973, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1973), (finish-start)*1e-9)
end