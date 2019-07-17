using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm69(ml2306::Array{Float64,2}, ml2307::Array{Float64,2}, ml2308::Array{Float64,2}, ml2309::Array{Float64,2}, ml2310::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2306, full, L: ml2307, full, A: ml2308, full, B: ml2309, full, y: ml2310, full
    ml2311 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2311, ml2309)

    # R: ml2306, full, L: ml2307, full, A: ml2308, full, y: ml2310, full, tmp26: ml2311, full
    ml2312 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2308, ml2312, info) = LinearAlgebra.LAPACK.getrf!(ml2308)

    # R: ml2306, full, L: ml2307, full, y: ml2310, full, tmp26: ml2311, full, P11: ml2312, ipiv, L9: ml2308, lower_triangular_udiag, U10: ml2308, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2308, ml2311)

    # R: ml2306, full, L: ml2307, full, y: ml2310, full, P11: ml2312, ipiv, L9: ml2308, lower_triangular_udiag, tmp27: ml2311, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2308, ml2311)

    # R: ml2306, full, L: ml2307, full, y: ml2310, full, P11: ml2312, ipiv, tmp28: ml2311, full
    ml2313 = [1:length(ml2312);]
    @inbounds for i in 1:length(ml2312)
        ml2313[i], ml2313[ml2312[i]] = ml2313[ml2312[i]], ml2313[i];
    end;
    ml2314 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2314 = ml2311[invperm(ml2313),:]

    # R: ml2306, full, L: ml2307, full, y: ml2310, full, tmp25: ml2314, full
    ml2315 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2314, 0.0, ml2315)

    # R: ml2306, full, L: ml2307, full, y: ml2310, full, tmp19: ml2315, symmetric_lower_triangular
    ml2316 = diag(ml2307)
    ml2317 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2306, 1, ml2317, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2306, 2);
        view(ml2306, :, i)[:] .*= ml2316;
    end;        

    # R: ml2317, full, y: ml2310, full, tmp19: ml2315, symmetric_lower_triangular, tmp29: ml2306, full
    for i = 1:2000-1;
        view(ml2315, i, i+1:2000)[:] = view(ml2315, i+1:2000, i);
    end;
    ml2318 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2315, 1, ml2318, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2306, ml2317, 1.0, ml2315)

    # y: ml2310, full, tmp19: ml2318, full, tmp31: ml2315, full
    ml2319 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2315, ml2319, info) = LinearAlgebra.LAPACK.getrf!(ml2315)

    # y: ml2310, full, tmp19: ml2318, full, P35: ml2319, ipiv, L33: ml2315, lower_triangular_udiag, U34: ml2315, upper_triangular
    ml2320 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2318, ml2310, 0.0, ml2320)

    # P35: ml2319, ipiv, L33: ml2315, lower_triangular_udiag, U34: ml2315, upper_triangular, tmp32: ml2320, full
    ml2321 = [1:length(ml2319);]
    @inbounds for i in 1:length(ml2319)
        ml2321[i], ml2321[ml2319[i]] = ml2321[ml2319[i]], ml2321[i];
    end;
    ml2322 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2322 = ml2320[ml2321]

    # L33: ml2315, lower_triangular_udiag, U34: ml2315, upper_triangular, tmp40: ml2322, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2315, ml2322)

    # U34: ml2315, upper_triangular, tmp41: ml2322, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2315, ml2322)

    # tmp17: ml2322, full
    # x = tmp17
    return (ml2322)
end