using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm97(ml3242::Array{Float64,2}, ml3243::Array{Float64,2}, ml3244::Array{Float64,2}, ml3245::Array{Float64,2}, ml3246::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3242, full, L: ml3243, full, A: ml3244, full, B: ml3245, full, y: ml3246, full
    ml3247 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3247, ml3245)

    # R: ml3242, full, L: ml3243, full, A: ml3244, full, y: ml3246, full, tmp26: ml3247, full
    ml3248 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3244, ml3248, info) = LinearAlgebra.LAPACK.getrf!(ml3244)

    # R: ml3242, full, L: ml3243, full, y: ml3246, full, tmp26: ml3247, full, P11: ml3248, ipiv, L9: ml3244, lower_triangular_udiag, U10: ml3244, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3244, ml3247)

    # R: ml3242, full, L: ml3243, full, y: ml3246, full, P11: ml3248, ipiv, L9: ml3244, lower_triangular_udiag, tmp27: ml3247, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3244, ml3247)

    # R: ml3242, full, L: ml3243, full, y: ml3246, full, P11: ml3248, ipiv, tmp28: ml3247, full
    ml3249 = [1:length(ml3248);]
    @inbounds for i in 1:length(ml3248)
        ml3249[i], ml3249[ml3248[i]] = ml3249[ml3248[i]], ml3249[i];
    end;
    ml3250 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3250 = ml3247[invperm(ml3249),:]

    # R: ml3242, full, L: ml3243, full, y: ml3246, full, tmp25: ml3250, full
    ml3251 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3250, 0.0, ml3251)

    # R: ml3242, full, L: ml3243, full, y: ml3246, full, tmp19: ml3251, symmetric_lower_triangular
    ml3252 = diag(ml3243)
    ml3253 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3242, 1, ml3253, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3242, 2);
        view(ml3242, :, i)[:] .*= ml3252;
    end;        

    # R: ml3253, full, y: ml3246, full, tmp19: ml3251, symmetric_lower_triangular, tmp29: ml3242, full
    for i = 1:2000-1;
        view(ml3251, i, i+1:2000)[:] = view(ml3251, i+1:2000, i);
    end;
    ml3254 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3251, 1, ml3254, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3242, ml3253, 1.0, ml3251)

    # y: ml3246, full, tmp19: ml3254, full, tmp31: ml3251, full
    ml3255 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3251, ml3255, info) = LinearAlgebra.LAPACK.getrf!(ml3251)

    # y: ml3246, full, tmp19: ml3254, full, P35: ml3255, ipiv, L33: ml3251, lower_triangular_udiag, U34: ml3251, upper_triangular
    ml3256 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3254, ml3246, 0.0, ml3256)

    # P35: ml3255, ipiv, L33: ml3251, lower_triangular_udiag, U34: ml3251, upper_triangular, tmp32: ml3256, full
    ml3257 = [1:length(ml3255);]
    @inbounds for i in 1:length(ml3255)
        ml3257[i], ml3257[ml3255[i]] = ml3257[ml3255[i]], ml3257[i];
    end;
    ml3258 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3258 = ml3256[ml3257]

    # L33: ml3251, lower_triangular_udiag, U34: ml3251, upper_triangular, tmp40: ml3258, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3251, ml3258)

    # U34: ml3251, upper_triangular, tmp41: ml3258, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3251, ml3258)

    # tmp17: ml3258, full
    # x = tmp17
    return (ml3258)
end