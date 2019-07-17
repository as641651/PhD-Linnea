using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm98(ml3276::Array{Float64,2}, ml3277::Array{Float64,2}, ml3278::Array{Float64,2}, ml3279::Array{Float64,2}, ml3280::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3276, full, L: ml3277, full, A: ml3278, full, B: ml3279, full, y: ml3280, full
    ml3281 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3281, ml3279)

    # R: ml3276, full, L: ml3277, full, A: ml3278, full, y: ml3280, full, tmp26: ml3281, full
    ml3282 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3278, ml3282, info) = LinearAlgebra.LAPACK.getrf!(ml3278)

    # R: ml3276, full, L: ml3277, full, y: ml3280, full, tmp26: ml3281, full, P11: ml3282, ipiv, L9: ml3278, lower_triangular_udiag, U10: ml3278, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3278, ml3281)

    # R: ml3276, full, L: ml3277, full, y: ml3280, full, P11: ml3282, ipiv, L9: ml3278, lower_triangular_udiag, tmp27: ml3281, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3278, ml3281)

    # R: ml3276, full, L: ml3277, full, y: ml3280, full, P11: ml3282, ipiv, tmp28: ml3281, full
    ml3283 = [1:length(ml3282);]
    @inbounds for i in 1:length(ml3282)
        ml3283[i], ml3283[ml3282[i]] = ml3283[ml3282[i]], ml3283[i];
    end;
    ml3284 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3284 = ml3281[invperm(ml3283),:]

    # R: ml3276, full, L: ml3277, full, y: ml3280, full, tmp25: ml3284, full
    ml3285 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3284, 0.0, ml3285)

    # R: ml3276, full, L: ml3277, full, y: ml3280, full, tmp19: ml3285, symmetric_lower_triangular
    ml3286 = diag(ml3277)
    ml3287 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3276, 1, ml3287, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3276, 2);
        view(ml3276, :, i)[:] .*= ml3286;
    end;        

    # R: ml3287, full, y: ml3280, full, tmp19: ml3285, symmetric_lower_triangular, tmp29: ml3276, full
    for i = 1:2000-1;
        view(ml3285, i, i+1:2000)[:] = view(ml3285, i+1:2000, i);
    end;
    ml3288 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3285, 1, ml3288, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3276, ml3287, 1.0, ml3285)

    # y: ml3280, full, tmp19: ml3288, full, tmp31: ml3285, full
    ml3289 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3285, ml3289, info) = LinearAlgebra.LAPACK.getrf!(ml3285)

    # y: ml3280, full, tmp19: ml3288, full, P35: ml3289, ipiv, L33: ml3285, lower_triangular_udiag, U34: ml3285, upper_triangular
    ml3290 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3288, ml3280, 0.0, ml3290)

    # P35: ml3289, ipiv, L33: ml3285, lower_triangular_udiag, U34: ml3285, upper_triangular, tmp32: ml3290, full
    ml3291 = [1:length(ml3289);]
    @inbounds for i in 1:length(ml3289)
        ml3291[i], ml3291[ml3289[i]] = ml3291[ml3289[i]], ml3291[i];
    end;
    ml3292 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3292 = ml3290[ml3291]

    # L33: ml3285, lower_triangular_udiag, U34: ml3285, upper_triangular, tmp40: ml3292, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3285, ml3292)

    # U34: ml3285, upper_triangular, tmp41: ml3292, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3285, ml3292)

    # tmp17: ml3292, full
    # x = tmp17
    return (ml3292)
end