using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm74(ml2493::Array{Float64,2}, ml2494::Array{Float64,2}, ml2495::Array{Float64,2}, ml2496::Array{Float64,2}, ml2497::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2493, full, L: ml2494, full, A: ml2495, full, B: ml2496, full, y: ml2497, full
    ml2498 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2498, ml2496)

    # R: ml2493, full, L: ml2494, full, A: ml2495, full, y: ml2497, full, tmp26: ml2498, full
    ml2499 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2495, ml2499, info) = LinearAlgebra.LAPACK.getrf!(ml2495)

    # R: ml2493, full, L: ml2494, full, y: ml2497, full, tmp26: ml2498, full, P11: ml2499, ipiv, L9: ml2495, lower_triangular_udiag, U10: ml2495, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2495, ml2498)

    # R: ml2493, full, L: ml2494, full, y: ml2497, full, P11: ml2499, ipiv, L9: ml2495, lower_triangular_udiag, tmp27: ml2498, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2495, ml2498)

    # R: ml2493, full, L: ml2494, full, y: ml2497, full, P11: ml2499, ipiv, tmp28: ml2498, full
    ml2500 = [1:length(ml2499);]
    @inbounds for i in 1:length(ml2499)
        ml2500[i], ml2500[ml2499[i]] = ml2500[ml2499[i]], ml2500[i];
    end;
    ml2501 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2501 = ml2498[invperm(ml2500),:]

    # R: ml2493, full, L: ml2494, full, y: ml2497, full, tmp25: ml2501, full
    ml2502 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2501, 0.0, ml2502)

    # R: ml2493, full, L: ml2494, full, y: ml2497, full, tmp19: ml2502, symmetric_lower_triangular
    ml2503 = diag(ml2494)
    ml2504 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2493, 1, ml2504, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2493, 2);
        view(ml2493, :, i)[:] .*= ml2503;
    end;        

    # R: ml2504, full, y: ml2497, full, tmp19: ml2502, symmetric_lower_triangular, tmp29: ml2493, full
    for i = 1:2000-1;
        view(ml2502, i, i+1:2000)[:] = view(ml2502, i+1:2000, i);
    end;
    ml2505 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2502, 1, ml2505, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2493, ml2504, 1.0, ml2502)

    # y: ml2497, full, tmp19: ml2505, full, tmp31: ml2502, full
    ml2506 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2502, ml2506, info) = LinearAlgebra.LAPACK.getrf!(ml2502)

    # y: ml2497, full, tmp19: ml2505, full, P35: ml2506, ipiv, L33: ml2502, lower_triangular_udiag, U34: ml2502, upper_triangular
    ml2507 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2505, ml2497, 0.0, ml2507)

    # P35: ml2506, ipiv, L33: ml2502, lower_triangular_udiag, U34: ml2502, upper_triangular, tmp32: ml2507, full
    ml2508 = [1:length(ml2506);]
    @inbounds for i in 1:length(ml2506)
        ml2508[i], ml2508[ml2506[i]] = ml2508[ml2506[i]], ml2508[i];
    end;
    ml2509 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2509 = ml2507[ml2508]

    # L33: ml2502, lower_triangular_udiag, U34: ml2502, upper_triangular, tmp40: ml2509, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2502, ml2509)

    # U34: ml2502, upper_triangular, tmp41: ml2509, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2502, ml2509)

    # tmp17: ml2509, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2509), (finish-start)*1e-9)
end