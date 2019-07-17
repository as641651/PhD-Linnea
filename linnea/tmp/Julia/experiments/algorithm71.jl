using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm71(ml2391::Array{Float64,2}, ml2392::Array{Float64,2}, ml2393::Array{Float64,2}, ml2394::Array{Float64,2}, ml2395::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2391, full, L: ml2392, full, A: ml2393, full, B: ml2394, full, y: ml2395, full
    ml2396 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2396, ml2394)

    # R: ml2391, full, L: ml2392, full, A: ml2393, full, y: ml2395, full, tmp26: ml2396, full
    ml2397 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2393, ml2397, info) = LinearAlgebra.LAPACK.getrf!(ml2393)

    # R: ml2391, full, L: ml2392, full, y: ml2395, full, tmp26: ml2396, full, P11: ml2397, ipiv, L9: ml2393, lower_triangular_udiag, U10: ml2393, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2393, ml2396)

    # R: ml2391, full, L: ml2392, full, y: ml2395, full, P11: ml2397, ipiv, L9: ml2393, lower_triangular_udiag, tmp27: ml2396, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2393, ml2396)

    # R: ml2391, full, L: ml2392, full, y: ml2395, full, P11: ml2397, ipiv, tmp28: ml2396, full
    ml2398 = [1:length(ml2397);]
    @inbounds for i in 1:length(ml2397)
        ml2398[i], ml2398[ml2397[i]] = ml2398[ml2397[i]], ml2398[i];
    end;
    ml2399 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2399 = ml2396[invperm(ml2398),:]

    # R: ml2391, full, L: ml2392, full, y: ml2395, full, tmp25: ml2399, full
    ml2400 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2399, 0.0, ml2400)

    # R: ml2391, full, L: ml2392, full, y: ml2395, full, tmp19: ml2400, symmetric_lower_triangular
    ml2401 = diag(ml2392)
    ml2402 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2391, 1, ml2402, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2391, 2);
        view(ml2391, :, i)[:] .*= ml2401;
    end;        

    # R: ml2402, full, y: ml2395, full, tmp19: ml2400, symmetric_lower_triangular, tmp29: ml2391, full
    for i = 1:2000-1;
        view(ml2400, i, i+1:2000)[:] = view(ml2400, i+1:2000, i);
    end;
    ml2403 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2400, 1, ml2403, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2391, ml2402, 1.0, ml2400)

    # y: ml2395, full, tmp19: ml2403, full, tmp31: ml2400, full
    ml2404 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2403, ml2395, 0.0, ml2404)

    # tmp31: ml2400, full, tmp32: ml2404, full
    ml2405 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2400, ml2405, info) = LinearAlgebra.LAPACK.getrf!(ml2400)

    # tmp32: ml2404, full, P35: ml2405, ipiv, L33: ml2400, lower_triangular_udiag, U34: ml2400, upper_triangular
    ml2406 = [1:length(ml2405);]
    @inbounds for i in 1:length(ml2405)
        ml2406[i], ml2406[ml2405[i]] = ml2406[ml2405[i]], ml2406[i];
    end;
    ml2407 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2407 = ml2404[ml2406]

    # L33: ml2400, lower_triangular_udiag, U34: ml2400, upper_triangular, tmp40: ml2407, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2400, ml2407)

    # U34: ml2400, upper_triangular, tmp41: ml2407, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2400, ml2407)

    # tmp17: ml2407, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2407), (finish-start)*1e-9)
end