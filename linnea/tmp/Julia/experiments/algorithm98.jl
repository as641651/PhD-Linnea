using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm98(ml3293::Array{Float64,2}, ml3294::Array{Float64,2}, ml3295::Array{Float64,2}, ml3296::Array{Float64,2}, ml3297::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3293, full, L: ml3294, full, A: ml3295, full, B: ml3296, full, y: ml3297, full
    ml3298 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3298, ml3296)

    # R: ml3293, full, L: ml3294, full, A: ml3295, full, y: ml3297, full, tmp26: ml3298, full
    ml3299 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3295, ml3299, info) = LinearAlgebra.LAPACK.getrf!(ml3295)

    # R: ml3293, full, L: ml3294, full, y: ml3297, full, tmp26: ml3298, full, P11: ml3299, ipiv, L9: ml3295, lower_triangular_udiag, U10: ml3295, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3295, ml3298)

    # R: ml3293, full, L: ml3294, full, y: ml3297, full, P11: ml3299, ipiv, L9: ml3295, lower_triangular_udiag, tmp27: ml3298, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3295, ml3298)

    # R: ml3293, full, L: ml3294, full, y: ml3297, full, P11: ml3299, ipiv, tmp28: ml3298, full
    ml3300 = [1:length(ml3299);]
    @inbounds for i in 1:length(ml3299)
        ml3300[i], ml3300[ml3299[i]] = ml3300[ml3299[i]], ml3300[i];
    end;
    ml3301 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3301 = ml3298[invperm(ml3300),:]

    # R: ml3293, full, L: ml3294, full, y: ml3297, full, tmp25: ml3301, full
    ml3302 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3301, 0.0, ml3302)

    # R: ml3293, full, L: ml3294, full, y: ml3297, full, tmp19: ml3302, symmetric_lower_triangular
    ml3303 = diag(ml3294)
    ml3304 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3293, 1, ml3304, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3293, 2);
        view(ml3293, :, i)[:] .*= ml3303;
    end;        

    # R: ml3304, full, y: ml3297, full, tmp19: ml3302, symmetric_lower_triangular, tmp29: ml3293, full
    for i = 1:2000-1;
        view(ml3302, i, i+1:2000)[:] = view(ml3302, i+1:2000, i);
    end;
    ml3305 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3302, 1, ml3305, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3293, ml3304, 1.0, ml3302)

    # y: ml3297, full, tmp19: ml3305, full, tmp31: ml3302, full
    ml3306 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3302, ml3306, info) = LinearAlgebra.LAPACK.getrf!(ml3302)

    # y: ml3297, full, tmp19: ml3305, full, P35: ml3306, ipiv, L33: ml3302, lower_triangular_udiag, U34: ml3302, upper_triangular
    ml3307 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3305, ml3297, 0.0, ml3307)

    # P35: ml3306, ipiv, L33: ml3302, lower_triangular_udiag, U34: ml3302, upper_triangular, tmp32: ml3307, full
    ml3308 = [1:length(ml3306);]
    @inbounds for i in 1:length(ml3306)
        ml3308[i], ml3308[ml3306[i]] = ml3308[ml3306[i]], ml3308[i];
    end;
    ml3309 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3309 = ml3307[ml3308]

    # L33: ml3302, lower_triangular_udiag, U34: ml3302, upper_triangular, tmp40: ml3309, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3302, ml3309)

    # U34: ml3302, upper_triangular, tmp41: ml3309, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3302, ml3309)

    # tmp17: ml3309, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3309), (finish-start)*1e-9)
end