using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm73(ml2459::Array{Float64,2}, ml2460::Array{Float64,2}, ml2461::Array{Float64,2}, ml2462::Array{Float64,2}, ml2463::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2459, full, L: ml2460, full, A: ml2461, full, B: ml2462, full, y: ml2463, full
    ml2464 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2464, ml2462)

    # R: ml2459, full, L: ml2460, full, A: ml2461, full, y: ml2463, full, tmp26: ml2464, full
    ml2465 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2461, ml2465, info) = LinearAlgebra.LAPACK.getrf!(ml2461)

    # R: ml2459, full, L: ml2460, full, y: ml2463, full, tmp26: ml2464, full, P11: ml2465, ipiv, L9: ml2461, lower_triangular_udiag, U10: ml2461, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2461, ml2464)

    # R: ml2459, full, L: ml2460, full, y: ml2463, full, P11: ml2465, ipiv, L9: ml2461, lower_triangular_udiag, tmp27: ml2464, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2461, ml2464)

    # R: ml2459, full, L: ml2460, full, y: ml2463, full, P11: ml2465, ipiv, tmp28: ml2464, full
    ml2466 = [1:length(ml2465);]
    @inbounds for i in 1:length(ml2465)
        ml2466[i], ml2466[ml2465[i]] = ml2466[ml2465[i]], ml2466[i];
    end;
    ml2467 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2467 = ml2464[invperm(ml2466),:]

    # R: ml2459, full, L: ml2460, full, y: ml2463, full, tmp25: ml2467, full
    ml2468 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2467, 0.0, ml2468)

    # R: ml2459, full, L: ml2460, full, y: ml2463, full, tmp19: ml2468, symmetric_lower_triangular
    ml2469 = diag(ml2460)
    ml2470 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2459, 1, ml2470, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2459, 2);
        view(ml2459, :, i)[:] .*= ml2469;
    end;        

    # R: ml2470, full, y: ml2463, full, tmp19: ml2468, symmetric_lower_triangular, tmp29: ml2459, full
    for i = 1:2000-1;
        view(ml2468, i, i+1:2000)[:] = view(ml2468, i+1:2000, i);
    end;
    ml2471 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2468, 1, ml2471, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2459, ml2470, 1.0, ml2468)

    # y: ml2463, full, tmp19: ml2471, full, tmp31: ml2468, full
    ml2472 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2471, ml2463, 0.0, ml2472)

    # tmp31: ml2468, full, tmp32: ml2472, full
    ml2473 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2468, ml2473, info) = LinearAlgebra.LAPACK.getrf!(ml2468)

    # tmp32: ml2472, full, P35: ml2473, ipiv, L33: ml2468, lower_triangular_udiag, U34: ml2468, upper_triangular
    ml2474 = [1:length(ml2473);]
    @inbounds for i in 1:length(ml2473)
        ml2474[i], ml2474[ml2473[i]] = ml2474[ml2473[i]], ml2474[i];
    end;
    ml2475 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2475 = ml2472[ml2474]

    # L33: ml2468, lower_triangular_udiag, U34: ml2468, upper_triangular, tmp40: ml2475, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2468, ml2475)

    # U34: ml2468, upper_triangular, tmp41: ml2475, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2468, ml2475)

    # tmp17: ml2475, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2475), (finish-start)*1e-9)
end