using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm73(ml2442::Array{Float64,2}, ml2443::Array{Float64,2}, ml2444::Array{Float64,2}, ml2445::Array{Float64,2}, ml2446::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2442, full, L: ml2443, full, A: ml2444, full, B: ml2445, full, y: ml2446, full
    ml2447 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2447, ml2445)

    # R: ml2442, full, L: ml2443, full, A: ml2444, full, y: ml2446, full, tmp26: ml2447, full
    ml2448 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2444, ml2448, info) = LinearAlgebra.LAPACK.getrf!(ml2444)

    # R: ml2442, full, L: ml2443, full, y: ml2446, full, tmp26: ml2447, full, P11: ml2448, ipiv, L9: ml2444, lower_triangular_udiag, U10: ml2444, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2444, ml2447)

    # R: ml2442, full, L: ml2443, full, y: ml2446, full, P11: ml2448, ipiv, L9: ml2444, lower_triangular_udiag, tmp27: ml2447, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2444, ml2447)

    # R: ml2442, full, L: ml2443, full, y: ml2446, full, P11: ml2448, ipiv, tmp28: ml2447, full
    ml2449 = [1:length(ml2448);]
    @inbounds for i in 1:length(ml2448)
        ml2449[i], ml2449[ml2448[i]] = ml2449[ml2448[i]], ml2449[i];
    end;
    ml2450 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2450 = ml2447[invperm(ml2449),:]

    # R: ml2442, full, L: ml2443, full, y: ml2446, full, tmp25: ml2450, full
    ml2451 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2450, 0.0, ml2451)

    # R: ml2442, full, L: ml2443, full, y: ml2446, full, tmp19: ml2451, symmetric_lower_triangular
    ml2452 = diag(ml2443)
    ml2453 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2442, 1, ml2453, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2442, 2);
        view(ml2442, :, i)[:] .*= ml2452;
    end;        

    # R: ml2453, full, y: ml2446, full, tmp19: ml2451, symmetric_lower_triangular, tmp29: ml2442, full
    for i = 1:2000-1;
        view(ml2451, i, i+1:2000)[:] = view(ml2451, i+1:2000, i);
    end;
    ml2454 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2451, 1, ml2454, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2442, ml2453, 1.0, ml2451)

    # y: ml2446, full, tmp19: ml2454, full, tmp31: ml2451, full
    ml2455 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2454, ml2446, 0.0, ml2455)

    # tmp31: ml2451, full, tmp32: ml2455, full
    ml2456 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2451, ml2456, info) = LinearAlgebra.LAPACK.getrf!(ml2451)

    # tmp32: ml2455, full, P35: ml2456, ipiv, L33: ml2451, lower_triangular_udiag, U34: ml2451, upper_triangular
    ml2457 = [1:length(ml2456);]
    @inbounds for i in 1:length(ml2456)
        ml2457[i], ml2457[ml2456[i]] = ml2457[ml2456[i]], ml2457[i];
    end;
    ml2458 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2458 = ml2455[ml2457]

    # L33: ml2451, lower_triangular_udiag, U34: ml2451, upper_triangular, tmp40: ml2458, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2451, ml2458)

    # U34: ml2451, upper_triangular, tmp41: ml2458, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2451, ml2458)

    # tmp17: ml2458, full
    # x = tmp17
    return (ml2458)
end