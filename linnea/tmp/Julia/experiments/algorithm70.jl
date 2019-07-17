using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm70(ml2357::Array{Float64,2}, ml2358::Array{Float64,2}, ml2359::Array{Float64,2}, ml2360::Array{Float64,2}, ml2361::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2357, full, L: ml2358, full, A: ml2359, full, B: ml2360, full, y: ml2361, full
    ml2362 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2362, ml2360)

    # R: ml2357, full, L: ml2358, full, A: ml2359, full, y: ml2361, full, tmp26: ml2362, full
    ml2363 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2359, ml2363, info) = LinearAlgebra.LAPACK.getrf!(ml2359)

    # R: ml2357, full, L: ml2358, full, y: ml2361, full, tmp26: ml2362, full, P11: ml2363, ipiv, L9: ml2359, lower_triangular_udiag, U10: ml2359, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2359, ml2362)

    # R: ml2357, full, L: ml2358, full, y: ml2361, full, P11: ml2363, ipiv, L9: ml2359, lower_triangular_udiag, tmp27: ml2362, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2359, ml2362)

    # R: ml2357, full, L: ml2358, full, y: ml2361, full, P11: ml2363, ipiv, tmp28: ml2362, full
    ml2364 = [1:length(ml2363);]
    @inbounds for i in 1:length(ml2363)
        ml2364[i], ml2364[ml2363[i]] = ml2364[ml2363[i]], ml2364[i];
    end;
    ml2365 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2365 = ml2362[invperm(ml2364),:]

    # R: ml2357, full, L: ml2358, full, y: ml2361, full, tmp25: ml2365, full
    ml2366 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2365, 0.0, ml2366)

    # R: ml2357, full, L: ml2358, full, y: ml2361, full, tmp19: ml2366, symmetric_lower_triangular
    ml2367 = diag(ml2358)
    ml2368 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2357, 1, ml2368, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2357, 2);
        view(ml2357, :, i)[:] .*= ml2367;
    end;        

    # R: ml2368, full, y: ml2361, full, tmp19: ml2366, symmetric_lower_triangular, tmp29: ml2357, full
    for i = 1:2000-1;
        view(ml2366, i, i+1:2000)[:] = view(ml2366, i+1:2000, i);
    end;
    ml2369 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2366, 1, ml2369, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2357, ml2368, 1.0, ml2366)

    # y: ml2361, full, tmp19: ml2369, full, tmp31: ml2366, full
    ml2370 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2369, ml2361, 0.0, ml2370)

    # tmp31: ml2366, full, tmp32: ml2370, full
    ml2371 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2366, ml2371, info) = LinearAlgebra.LAPACK.getrf!(ml2366)

    # tmp32: ml2370, full, P35: ml2371, ipiv, L33: ml2366, lower_triangular_udiag, U34: ml2366, upper_triangular
    ml2372 = [1:length(ml2371);]
    @inbounds for i in 1:length(ml2371)
        ml2372[i], ml2372[ml2371[i]] = ml2372[ml2371[i]], ml2372[i];
    end;
    ml2373 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2373 = ml2370[ml2372]

    # L33: ml2366, lower_triangular_udiag, U34: ml2366, upper_triangular, tmp40: ml2373, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2366, ml2373)

    # U34: ml2366, upper_triangular, tmp41: ml2373, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2366, ml2373)

    # tmp17: ml2373, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2373), (finish-start)*1e-9)
end