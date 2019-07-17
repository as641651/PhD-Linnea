using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm95(ml3174::Array{Float64,2}, ml3175::Array{Float64,2}, ml3176::Array{Float64,2}, ml3177::Array{Float64,2}, ml3178::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3174, full, L: ml3175, full, A: ml3176, full, B: ml3177, full, y: ml3178, full
    ml3179 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3179, ml3177)

    # R: ml3174, full, L: ml3175, full, A: ml3176, full, y: ml3178, full, tmp26: ml3179, full
    ml3180 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3176, ml3180, info) = LinearAlgebra.LAPACK.getrf!(ml3176)

    # R: ml3174, full, L: ml3175, full, y: ml3178, full, tmp26: ml3179, full, P11: ml3180, ipiv, L9: ml3176, lower_triangular_udiag, U10: ml3176, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3176, ml3179)

    # R: ml3174, full, L: ml3175, full, y: ml3178, full, P11: ml3180, ipiv, L9: ml3176, lower_triangular_udiag, tmp27: ml3179, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3176, ml3179)

    # R: ml3174, full, L: ml3175, full, y: ml3178, full, P11: ml3180, ipiv, tmp28: ml3179, full
    ml3181 = [1:length(ml3180);]
    @inbounds for i in 1:length(ml3180)
        ml3181[i], ml3181[ml3180[i]] = ml3181[ml3180[i]], ml3181[i];
    end;
    ml3182 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3182 = ml3179[invperm(ml3181),:]

    # R: ml3174, full, L: ml3175, full, y: ml3178, full, tmp25: ml3182, full
    ml3183 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3182, 0.0, ml3183)

    # R: ml3174, full, L: ml3175, full, y: ml3178, full, tmp19: ml3183, symmetric_lower_triangular
    ml3184 = diag(ml3175)
    ml3185 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3174, 1, ml3185, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3174, 2);
        view(ml3174, :, i)[:] .*= ml3184;
    end;        

    # R: ml3185, full, y: ml3178, full, tmp19: ml3183, symmetric_lower_triangular, tmp29: ml3174, full
    for i = 1:2000-1;
        view(ml3183, i, i+1:2000)[:] = view(ml3183, i+1:2000, i);
    end;
    ml3186 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3183, 1, ml3186, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3174, ml3185, 1.0, ml3183)

    # y: ml3178, full, tmp19: ml3186, full, tmp31: ml3183, full
    ml3187 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3183, ml3187, info) = LinearAlgebra.LAPACK.getrf!(ml3183)

    # y: ml3178, full, tmp19: ml3186, full, P35: ml3187, ipiv, L33: ml3183, lower_triangular_udiag, U34: ml3183, upper_triangular
    ml3188 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3186, ml3178, 0.0, ml3188)

    # P35: ml3187, ipiv, L33: ml3183, lower_triangular_udiag, U34: ml3183, upper_triangular, tmp32: ml3188, full
    ml3189 = [1:length(ml3187);]
    @inbounds for i in 1:length(ml3187)
        ml3189[i], ml3189[ml3187[i]] = ml3189[ml3187[i]], ml3189[i];
    end;
    ml3190 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3190 = ml3188[ml3189]

    # L33: ml3183, lower_triangular_udiag, U34: ml3183, upper_triangular, tmp40: ml3190, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3183, ml3190)

    # U34: ml3183, upper_triangular, tmp41: ml3190, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3183, ml3190)

    # tmp17: ml3190, full
    # x = tmp17
    return (ml3190)
end