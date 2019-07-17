using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm99(ml3327::Array{Float64,2}, ml3328::Array{Float64,2}, ml3329::Array{Float64,2}, ml3330::Array{Float64,2}, ml3331::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3327, full, L: ml3328, full, A: ml3329, full, B: ml3330, full, y: ml3331, full
    ml3332 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3332, ml3330)

    # R: ml3327, full, L: ml3328, full, A: ml3329, full, y: ml3331, full, tmp26: ml3332, full
    ml3333 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3329, ml3333, info) = LinearAlgebra.LAPACK.getrf!(ml3329)

    # R: ml3327, full, L: ml3328, full, y: ml3331, full, tmp26: ml3332, full, P11: ml3333, ipiv, L9: ml3329, lower_triangular_udiag, U10: ml3329, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3329, ml3332)

    # R: ml3327, full, L: ml3328, full, y: ml3331, full, P11: ml3333, ipiv, L9: ml3329, lower_triangular_udiag, tmp27: ml3332, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3329, ml3332)

    # R: ml3327, full, L: ml3328, full, y: ml3331, full, P11: ml3333, ipiv, tmp28: ml3332, full
    ml3334 = [1:length(ml3333);]
    @inbounds for i in 1:length(ml3333)
        ml3334[i], ml3334[ml3333[i]] = ml3334[ml3333[i]], ml3334[i];
    end;
    ml3335 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3335 = ml3332[invperm(ml3334),:]

    # R: ml3327, full, L: ml3328, full, y: ml3331, full, tmp25: ml3335, full
    ml3336 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3335, 0.0, ml3336)

    # R: ml3327, full, L: ml3328, full, y: ml3331, full, tmp19: ml3336, symmetric_lower_triangular
    ml3337 = diag(ml3328)
    ml3338 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3327, 1, ml3338, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3327, 2);
        view(ml3327, :, i)[:] .*= ml3337;
    end;        

    # R: ml3338, full, y: ml3331, full, tmp19: ml3336, symmetric_lower_triangular, tmp29: ml3327, full
    for i = 1:2000-1;
        view(ml3336, i, i+1:2000)[:] = view(ml3336, i+1:2000, i);
    end;
    ml3339 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3336, 1, ml3339, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3327, ml3338, 1.0, ml3336)

    # y: ml3331, full, tmp19: ml3339, full, tmp31: ml3336, full
    ml3340 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3336, ml3340, info) = LinearAlgebra.LAPACK.getrf!(ml3336)

    # y: ml3331, full, tmp19: ml3339, full, P35: ml3340, ipiv, L33: ml3336, lower_triangular_udiag, U34: ml3336, upper_triangular
    ml3341 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3339, ml3331, 0.0, ml3341)

    # P35: ml3340, ipiv, L33: ml3336, lower_triangular_udiag, U34: ml3336, upper_triangular, tmp32: ml3341, full
    ml3342 = [1:length(ml3340);]
    @inbounds for i in 1:length(ml3340)
        ml3342[i], ml3342[ml3340[i]] = ml3342[ml3340[i]], ml3342[i];
    end;
    ml3343 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3343 = ml3341[ml3342]

    # L33: ml3336, lower_triangular_udiag, U34: ml3336, upper_triangular, tmp40: ml3343, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3336, ml3343)

    # U34: ml3336, upper_triangular, tmp41: ml3343, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3336, ml3343)

    # tmp17: ml3343, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3343), (finish-start)*1e-9)
end