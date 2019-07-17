using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm69(ml2323::Array{Float64,2}, ml2324::Array{Float64,2}, ml2325::Array{Float64,2}, ml2326::Array{Float64,2}, ml2327::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2323, full, L: ml2324, full, A: ml2325, full, B: ml2326, full, y: ml2327, full
    ml2328 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2328, ml2326)

    # R: ml2323, full, L: ml2324, full, A: ml2325, full, y: ml2327, full, tmp26: ml2328, full
    ml2329 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2325, ml2329, info) = LinearAlgebra.LAPACK.getrf!(ml2325)

    # R: ml2323, full, L: ml2324, full, y: ml2327, full, tmp26: ml2328, full, P11: ml2329, ipiv, L9: ml2325, lower_triangular_udiag, U10: ml2325, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2325, ml2328)

    # R: ml2323, full, L: ml2324, full, y: ml2327, full, P11: ml2329, ipiv, L9: ml2325, lower_triangular_udiag, tmp27: ml2328, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2325, ml2328)

    # R: ml2323, full, L: ml2324, full, y: ml2327, full, P11: ml2329, ipiv, tmp28: ml2328, full
    ml2330 = [1:length(ml2329);]
    @inbounds for i in 1:length(ml2329)
        ml2330[i], ml2330[ml2329[i]] = ml2330[ml2329[i]], ml2330[i];
    end;
    ml2331 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2331 = ml2328[invperm(ml2330),:]

    # R: ml2323, full, L: ml2324, full, y: ml2327, full, tmp25: ml2331, full
    ml2332 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2331, 0.0, ml2332)

    # R: ml2323, full, L: ml2324, full, y: ml2327, full, tmp19: ml2332, symmetric_lower_triangular
    ml2333 = diag(ml2324)
    ml2334 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2323, 1, ml2334, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2323, 2);
        view(ml2323, :, i)[:] .*= ml2333;
    end;        

    # R: ml2334, full, y: ml2327, full, tmp19: ml2332, symmetric_lower_triangular, tmp29: ml2323, full
    for i = 1:2000-1;
        view(ml2332, i, i+1:2000)[:] = view(ml2332, i+1:2000, i);
    end;
    ml2335 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2332, 1, ml2335, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2323, ml2334, 1.0, ml2332)

    # y: ml2327, full, tmp19: ml2335, full, tmp31: ml2332, full
    ml2336 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2332, ml2336, info) = LinearAlgebra.LAPACK.getrf!(ml2332)

    # y: ml2327, full, tmp19: ml2335, full, P35: ml2336, ipiv, L33: ml2332, lower_triangular_udiag, U34: ml2332, upper_triangular
    ml2337 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2335, ml2327, 0.0, ml2337)

    # P35: ml2336, ipiv, L33: ml2332, lower_triangular_udiag, U34: ml2332, upper_triangular, tmp32: ml2337, full
    ml2338 = [1:length(ml2336);]
    @inbounds for i in 1:length(ml2336)
        ml2338[i], ml2338[ml2336[i]] = ml2338[ml2336[i]], ml2338[i];
    end;
    ml2339 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2339 = ml2337[ml2338]

    # L33: ml2332, lower_triangular_udiag, U34: ml2332, upper_triangular, tmp40: ml2339, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2332, ml2339)

    # U34: ml2332, upper_triangular, tmp41: ml2339, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2332, ml2339)

    # tmp17: ml2339, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2339), (finish-start)*1e-9)
end