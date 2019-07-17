using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm95(ml3191::Array{Float64,2}, ml3192::Array{Float64,2}, ml3193::Array{Float64,2}, ml3194::Array{Float64,2}, ml3195::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3191, full, L: ml3192, full, A: ml3193, full, B: ml3194, full, y: ml3195, full
    ml3196 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3196, ml3194)

    # R: ml3191, full, L: ml3192, full, A: ml3193, full, y: ml3195, full, tmp26: ml3196, full
    ml3197 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3193, ml3197, info) = LinearAlgebra.LAPACK.getrf!(ml3193)

    # R: ml3191, full, L: ml3192, full, y: ml3195, full, tmp26: ml3196, full, P11: ml3197, ipiv, L9: ml3193, lower_triangular_udiag, U10: ml3193, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3193, ml3196)

    # R: ml3191, full, L: ml3192, full, y: ml3195, full, P11: ml3197, ipiv, L9: ml3193, lower_triangular_udiag, tmp27: ml3196, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3193, ml3196)

    # R: ml3191, full, L: ml3192, full, y: ml3195, full, P11: ml3197, ipiv, tmp28: ml3196, full
    ml3198 = [1:length(ml3197);]
    @inbounds for i in 1:length(ml3197)
        ml3198[i], ml3198[ml3197[i]] = ml3198[ml3197[i]], ml3198[i];
    end;
    ml3199 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3199 = ml3196[invperm(ml3198),:]

    # R: ml3191, full, L: ml3192, full, y: ml3195, full, tmp25: ml3199, full
    ml3200 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3199, 0.0, ml3200)

    # R: ml3191, full, L: ml3192, full, y: ml3195, full, tmp19: ml3200, symmetric_lower_triangular
    ml3201 = diag(ml3192)
    ml3202 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3191, 1, ml3202, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3191, 2);
        view(ml3191, :, i)[:] .*= ml3201;
    end;        

    # R: ml3202, full, y: ml3195, full, tmp19: ml3200, symmetric_lower_triangular, tmp29: ml3191, full
    for i = 1:2000-1;
        view(ml3200, i, i+1:2000)[:] = view(ml3200, i+1:2000, i);
    end;
    ml3203 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3200, 1, ml3203, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3191, ml3202, 1.0, ml3200)

    # y: ml3195, full, tmp19: ml3203, full, tmp31: ml3200, full
    ml3204 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3200, ml3204, info) = LinearAlgebra.LAPACK.getrf!(ml3200)

    # y: ml3195, full, tmp19: ml3203, full, P35: ml3204, ipiv, L33: ml3200, lower_triangular_udiag, U34: ml3200, upper_triangular
    ml3205 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3203, ml3195, 0.0, ml3205)

    # P35: ml3204, ipiv, L33: ml3200, lower_triangular_udiag, U34: ml3200, upper_triangular, tmp32: ml3205, full
    ml3206 = [1:length(ml3204);]
    @inbounds for i in 1:length(ml3204)
        ml3206[i], ml3206[ml3204[i]] = ml3206[ml3204[i]], ml3206[i];
    end;
    ml3207 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3207 = ml3205[ml3206]

    # L33: ml3200, lower_triangular_udiag, U34: ml3200, upper_triangular, tmp40: ml3207, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3200, ml3207)

    # U34: ml3200, upper_triangular, tmp41: ml3207, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3200, ml3207)

    # tmp17: ml3207, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3207), (finish-start)*1e-9)
end