using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm71(ml2374::Array{Float64,2}, ml2375::Array{Float64,2}, ml2376::Array{Float64,2}, ml2377::Array{Float64,2}, ml2378::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2374, full, L: ml2375, full, A: ml2376, full, B: ml2377, full, y: ml2378, full
    ml2379 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2379, ml2377)

    # R: ml2374, full, L: ml2375, full, A: ml2376, full, y: ml2378, full, tmp26: ml2379, full
    ml2380 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2376, ml2380, info) = LinearAlgebra.LAPACK.getrf!(ml2376)

    # R: ml2374, full, L: ml2375, full, y: ml2378, full, tmp26: ml2379, full, P11: ml2380, ipiv, L9: ml2376, lower_triangular_udiag, U10: ml2376, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2376, ml2379)

    # R: ml2374, full, L: ml2375, full, y: ml2378, full, P11: ml2380, ipiv, L9: ml2376, lower_triangular_udiag, tmp27: ml2379, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2376, ml2379)

    # R: ml2374, full, L: ml2375, full, y: ml2378, full, P11: ml2380, ipiv, tmp28: ml2379, full
    ml2381 = [1:length(ml2380);]
    @inbounds for i in 1:length(ml2380)
        ml2381[i], ml2381[ml2380[i]] = ml2381[ml2380[i]], ml2381[i];
    end;
    ml2382 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2382 = ml2379[invperm(ml2381),:]

    # R: ml2374, full, L: ml2375, full, y: ml2378, full, tmp25: ml2382, full
    ml2383 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2382, 0.0, ml2383)

    # R: ml2374, full, L: ml2375, full, y: ml2378, full, tmp19: ml2383, symmetric_lower_triangular
    ml2384 = diag(ml2375)
    ml2385 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2374, 1, ml2385, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2374, 2);
        view(ml2374, :, i)[:] .*= ml2384;
    end;        

    # R: ml2385, full, y: ml2378, full, tmp19: ml2383, symmetric_lower_triangular, tmp29: ml2374, full
    for i = 1:2000-1;
        view(ml2383, i, i+1:2000)[:] = view(ml2383, i+1:2000, i);
    end;
    ml2386 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2383, 1, ml2386, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2374, ml2385, 1.0, ml2383)

    # y: ml2378, full, tmp19: ml2386, full, tmp31: ml2383, full
    ml2387 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2386, ml2378, 0.0, ml2387)

    # tmp31: ml2383, full, tmp32: ml2387, full
    ml2388 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2383, ml2388, info) = LinearAlgebra.LAPACK.getrf!(ml2383)

    # tmp32: ml2387, full, P35: ml2388, ipiv, L33: ml2383, lower_triangular_udiag, U34: ml2383, upper_triangular
    ml2389 = [1:length(ml2388);]
    @inbounds for i in 1:length(ml2388)
        ml2389[i], ml2389[ml2388[i]] = ml2389[ml2388[i]], ml2389[i];
    end;
    ml2390 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2390 = ml2387[ml2389]

    # L33: ml2383, lower_triangular_udiag, U34: ml2383, upper_triangular, tmp40: ml2390, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2383, ml2390)

    # U34: ml2383, upper_triangular, tmp41: ml2390, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2383, ml2390)

    # tmp17: ml2390, full
    # x = tmp17
    return (ml2390)
end