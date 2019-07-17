using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm72(ml2425::Array{Float64,2}, ml2426::Array{Float64,2}, ml2427::Array{Float64,2}, ml2428::Array{Float64,2}, ml2429::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2425, full, L: ml2426, full, A: ml2427, full, B: ml2428, full, y: ml2429, full
    ml2430 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2430, ml2428)

    # R: ml2425, full, L: ml2426, full, A: ml2427, full, y: ml2429, full, tmp26: ml2430, full
    ml2431 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2427, ml2431, info) = LinearAlgebra.LAPACK.getrf!(ml2427)

    # R: ml2425, full, L: ml2426, full, y: ml2429, full, tmp26: ml2430, full, P11: ml2431, ipiv, L9: ml2427, lower_triangular_udiag, U10: ml2427, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2427, ml2430)

    # R: ml2425, full, L: ml2426, full, y: ml2429, full, P11: ml2431, ipiv, L9: ml2427, lower_triangular_udiag, tmp27: ml2430, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2427, ml2430)

    # R: ml2425, full, L: ml2426, full, y: ml2429, full, P11: ml2431, ipiv, tmp28: ml2430, full
    ml2432 = [1:length(ml2431);]
    @inbounds for i in 1:length(ml2431)
        ml2432[i], ml2432[ml2431[i]] = ml2432[ml2431[i]], ml2432[i];
    end;
    ml2433 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2433 = ml2430[invperm(ml2432),:]

    # R: ml2425, full, L: ml2426, full, y: ml2429, full, tmp25: ml2433, full
    ml2434 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2433, 0.0, ml2434)

    # R: ml2425, full, L: ml2426, full, y: ml2429, full, tmp19: ml2434, symmetric_lower_triangular
    ml2435 = diag(ml2426)
    ml2436 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2425, 1, ml2436, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2425, 2);
        view(ml2425, :, i)[:] .*= ml2435;
    end;        

    # R: ml2436, full, y: ml2429, full, tmp19: ml2434, symmetric_lower_triangular, tmp29: ml2425, full
    for i = 1:2000-1;
        view(ml2434, i, i+1:2000)[:] = view(ml2434, i+1:2000, i);
    end;
    ml2437 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2434, 1, ml2437, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2425, ml2436, 1.0, ml2434)

    # y: ml2429, full, tmp19: ml2437, full, tmp31: ml2434, full
    ml2438 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2437, ml2429, 0.0, ml2438)

    # tmp31: ml2434, full, tmp32: ml2438, full
    ml2439 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2434, ml2439, info) = LinearAlgebra.LAPACK.getrf!(ml2434)

    # tmp32: ml2438, full, P35: ml2439, ipiv, L33: ml2434, lower_triangular_udiag, U34: ml2434, upper_triangular
    ml2440 = [1:length(ml2439);]
    @inbounds for i in 1:length(ml2439)
        ml2440[i], ml2440[ml2439[i]] = ml2440[ml2439[i]], ml2440[i];
    end;
    ml2441 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2441 = ml2438[ml2440]

    # L33: ml2434, lower_triangular_udiag, U34: ml2434, upper_triangular, tmp40: ml2441, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2434, ml2441)

    # U34: ml2434, upper_triangular, tmp41: ml2441, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2434, ml2441)

    # tmp17: ml2441, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2441), (finish-start)*1e-9)
end