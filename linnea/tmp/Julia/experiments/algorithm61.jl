using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm61(ml2056::Array{Float64,2}, ml2057::Array{Float64,2}, ml2058::Array{Float64,2}, ml2059::Array{Float64,2}, ml2060::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2056, full, L: ml2057, full, A: ml2058, full, B: ml2059, full, y: ml2060, full
    ml2061 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2061, ml2059)

    # R: ml2056, full, L: ml2057, full, A: ml2058, full, y: ml2060, full, tmp26: ml2061, full
    ml2062 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2058, ml2062, info) = LinearAlgebra.LAPACK.getrf!(ml2058)

    # R: ml2056, full, L: ml2057, full, y: ml2060, full, tmp26: ml2061, full, P11: ml2062, ipiv, L9: ml2058, lower_triangular_udiag, U10: ml2058, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2058, ml2061)

    # R: ml2056, full, L: ml2057, full, y: ml2060, full, P11: ml2062, ipiv, L9: ml2058, lower_triangular_udiag, tmp27: ml2061, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2058, ml2061)

    # R: ml2056, full, L: ml2057, full, y: ml2060, full, P11: ml2062, ipiv, tmp28: ml2061, full
    ml2063 = [1:length(ml2062);]
    @inbounds for i in 1:length(ml2062)
        ml2063[i], ml2063[ml2062[i]] = ml2063[ml2062[i]], ml2063[i];
    end;
    ml2064 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2064 = ml2061[invperm(ml2063),:]

    # R: ml2056, full, L: ml2057, full, y: ml2060, full, tmp25: ml2064, full
    ml2065 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2064, 0.0, ml2065)

    # R: ml2056, full, L: ml2057, full, y: ml2060, full, tmp19: ml2065, symmetric_lower_triangular
    ml2066 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2065, ml2060, 0.0, ml2066)

    # R: ml2056, full, L: ml2057, full, tmp19: ml2065, symmetric_lower_triangular, tmp32: ml2066, full
    ml2067 = diag(ml2057)
    ml2068 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2056, 1, ml2068, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2056, 2);
        view(ml2056, :, i)[:] .*= ml2067;
    end;        

    # R: ml2068, full, tmp19: ml2065, symmetric_lower_triangular, tmp32: ml2066, full, tmp29: ml2056, full
    for i = 1:2000-1;
        view(ml2065, i, i+1:2000)[:] = view(ml2065, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2068, ml2056, 1.0, ml2065)

    # tmp32: ml2066, full, tmp31: ml2065, full
    ml2069 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2065, ml2069, info) = LinearAlgebra.LAPACK.getrf!(ml2065)

    # tmp32: ml2066, full, P35: ml2069, ipiv, L33: ml2065, lower_triangular_udiag, U34: ml2065, upper_triangular
    ml2070 = [1:length(ml2069);]
    @inbounds for i in 1:length(ml2069)
        ml2070[i], ml2070[ml2069[i]] = ml2070[ml2069[i]], ml2070[i];
    end;
    ml2071 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2071 = ml2066[ml2070]

    # L33: ml2065, lower_triangular_udiag, U34: ml2065, upper_triangular, tmp40: ml2071, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2065, ml2071)

    # U34: ml2065, upper_triangular, tmp41: ml2071, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2065, ml2071)

    # tmp17: ml2071, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2071), (finish-start)*1e-9)
end