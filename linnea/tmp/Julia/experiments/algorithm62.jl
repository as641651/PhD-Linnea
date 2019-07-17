using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm62(ml2088::Array{Float64,2}, ml2089::Array{Float64,2}, ml2090::Array{Float64,2}, ml2091::Array{Float64,2}, ml2092::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2088, full, L: ml2089, full, A: ml2090, full, B: ml2091, full, y: ml2092, full
    ml2093 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2093, ml2091)

    # R: ml2088, full, L: ml2089, full, A: ml2090, full, y: ml2092, full, tmp26: ml2093, full
    ml2094 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2090, ml2094, info) = LinearAlgebra.LAPACK.getrf!(ml2090)

    # R: ml2088, full, L: ml2089, full, y: ml2092, full, tmp26: ml2093, full, P11: ml2094, ipiv, L9: ml2090, lower_triangular_udiag, U10: ml2090, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2090, ml2093)

    # R: ml2088, full, L: ml2089, full, y: ml2092, full, P11: ml2094, ipiv, L9: ml2090, lower_triangular_udiag, tmp27: ml2093, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2090, ml2093)

    # R: ml2088, full, L: ml2089, full, y: ml2092, full, P11: ml2094, ipiv, tmp28: ml2093, full
    ml2095 = [1:length(ml2094);]
    @inbounds for i in 1:length(ml2094)
        ml2095[i], ml2095[ml2094[i]] = ml2095[ml2094[i]], ml2095[i];
    end;
    ml2096 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2096 = ml2093[invperm(ml2095),:]

    # R: ml2088, full, L: ml2089, full, y: ml2092, full, tmp25: ml2096, full
    ml2097 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2096, 0.0, ml2097)

    # R: ml2088, full, L: ml2089, full, y: ml2092, full, tmp19: ml2097, symmetric_lower_triangular
    ml2098 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2097, ml2092, 0.0, ml2098)

    # R: ml2088, full, L: ml2089, full, tmp19: ml2097, symmetric_lower_triangular, tmp32: ml2098, full
    ml2099 = diag(ml2089)
    ml2100 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2088, 1, ml2100, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2088, 2);
        view(ml2088, :, i)[:] .*= ml2099;
    end;        

    # R: ml2100, full, tmp19: ml2097, symmetric_lower_triangular, tmp32: ml2098, full, tmp29: ml2088, full
    for i = 1:2000-1;
        view(ml2097, i, i+1:2000)[:] = view(ml2097, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2100, ml2088, 1.0, ml2097)

    # tmp32: ml2098, full, tmp31: ml2097, full
    ml2101 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2097, ml2101, info) = LinearAlgebra.LAPACK.getrf!(ml2097)

    # tmp32: ml2098, full, P35: ml2101, ipiv, L33: ml2097, lower_triangular_udiag, U34: ml2097, upper_triangular
    ml2102 = [1:length(ml2101);]
    @inbounds for i in 1:length(ml2101)
        ml2102[i], ml2102[ml2101[i]] = ml2102[ml2101[i]], ml2102[i];
    end;
    ml2103 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2103 = ml2098[ml2102]

    # L33: ml2097, lower_triangular_udiag, U34: ml2097, upper_triangular, tmp40: ml2103, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2097, ml2103)

    # U34: ml2097, upper_triangular, tmp41: ml2103, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2097, ml2103)

    # tmp17: ml2103, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2103), (finish-start)*1e-9)
end