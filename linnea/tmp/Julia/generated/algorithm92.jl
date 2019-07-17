using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm92(ml3076::Array{Float64,2}, ml3077::Array{Float64,2}, ml3078::Array{Float64,2}, ml3079::Array{Float64,2}, ml3080::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3076, full, L: ml3077, full, A: ml3078, full, B: ml3079, full, y: ml3080, full
    ml3081 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3081, ml3079)

    # R: ml3076, full, L: ml3077, full, A: ml3078, full, y: ml3080, full, tmp26: ml3081, full
    ml3082 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3078, ml3082, info) = LinearAlgebra.LAPACK.getrf!(ml3078)

    # R: ml3076, full, L: ml3077, full, y: ml3080, full, tmp26: ml3081, full, P11: ml3082, ipiv, L9: ml3078, lower_triangular_udiag, U10: ml3078, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3078, ml3081)

    # R: ml3076, full, L: ml3077, full, y: ml3080, full, P11: ml3082, ipiv, L9: ml3078, lower_triangular_udiag, tmp27: ml3081, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3078, ml3081)

    # R: ml3076, full, L: ml3077, full, y: ml3080, full, P11: ml3082, ipiv, tmp28: ml3081, full
    ml3083 = [1:length(ml3082);]
    @inbounds for i in 1:length(ml3082)
        ml3083[i], ml3083[ml3082[i]] = ml3083[ml3082[i]], ml3083[i];
    end;
    ml3084 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3084 = ml3081[invperm(ml3083),:]

    # R: ml3076, full, L: ml3077, full, y: ml3080, full, tmp25: ml3084, full
    ml3085 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3084, 0.0, ml3085)

    # R: ml3076, full, L: ml3077, full, y: ml3080, full, tmp19: ml3085, symmetric_lower_triangular
    ml3086 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3085, ml3080, 0.0, ml3086)

    # R: ml3076, full, L: ml3077, full, tmp19: ml3085, symmetric_lower_triangular, tmp32: ml3086, full
    ml3087 = diag(ml3077)
    ml3088 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3076, 1, ml3088, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3076, 2);
        view(ml3076, :, i)[:] .*= ml3087;
    end;        

    # R: ml3088, full, tmp19: ml3085, symmetric_lower_triangular, tmp32: ml3086, full, tmp29: ml3076, full
    for i = 1:2000-1;
        view(ml3085, i, i+1:2000)[:] = view(ml3085, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3088, ml3076, 1.0, ml3085)

    # tmp32: ml3086, full, tmp31: ml3085, full
    ml3089 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3085, ml3089, info) = LinearAlgebra.LAPACK.getrf!(ml3085)

    # tmp32: ml3086, full, P35: ml3089, ipiv, L33: ml3085, lower_triangular_udiag, U34: ml3085, upper_triangular
    ml3090 = [1:length(ml3089);]
    @inbounds for i in 1:length(ml3089)
        ml3090[i], ml3090[ml3089[i]] = ml3090[ml3089[i]], ml3090[i];
    end;
    ml3091 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3091 = ml3086[ml3090]

    # L33: ml3085, lower_triangular_udiag, U34: ml3085, upper_triangular, tmp40: ml3091, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3085, ml3091)

    # U34: ml3085, upper_triangular, tmp41: ml3091, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3085, ml3091)

    # tmp17: ml3091, full
    # x = tmp17
    return (ml3091)
end