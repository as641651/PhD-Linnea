using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm93(ml3108::Array{Float64,2}, ml3109::Array{Float64,2}, ml3110::Array{Float64,2}, ml3111::Array{Float64,2}, ml3112::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3108, full, L: ml3109, full, A: ml3110, full, B: ml3111, full, y: ml3112, full
    ml3113 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3113, ml3111)

    # R: ml3108, full, L: ml3109, full, A: ml3110, full, y: ml3112, full, tmp26: ml3113, full
    ml3114 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3110, ml3114, info) = LinearAlgebra.LAPACK.getrf!(ml3110)

    # R: ml3108, full, L: ml3109, full, y: ml3112, full, tmp26: ml3113, full, P11: ml3114, ipiv, L9: ml3110, lower_triangular_udiag, U10: ml3110, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3110, ml3113)

    # R: ml3108, full, L: ml3109, full, y: ml3112, full, P11: ml3114, ipiv, L9: ml3110, lower_triangular_udiag, tmp27: ml3113, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3110, ml3113)

    # R: ml3108, full, L: ml3109, full, y: ml3112, full, P11: ml3114, ipiv, tmp28: ml3113, full
    ml3115 = [1:length(ml3114);]
    @inbounds for i in 1:length(ml3114)
        ml3115[i], ml3115[ml3114[i]] = ml3115[ml3114[i]], ml3115[i];
    end;
    ml3116 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3116 = ml3113[invperm(ml3115),:]

    # R: ml3108, full, L: ml3109, full, y: ml3112, full, tmp25: ml3116, full
    ml3117 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3116, 0.0, ml3117)

    # R: ml3108, full, L: ml3109, full, y: ml3112, full, tmp19: ml3117, symmetric_lower_triangular
    ml3118 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3117, ml3112, 0.0, ml3118)

    # R: ml3108, full, L: ml3109, full, tmp19: ml3117, symmetric_lower_triangular, tmp32: ml3118, full
    ml3119 = diag(ml3109)
    ml3120 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3108, 1, ml3120, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3108, 2);
        view(ml3108, :, i)[:] .*= ml3119;
    end;        

    # R: ml3120, full, tmp19: ml3117, symmetric_lower_triangular, tmp32: ml3118, full, tmp29: ml3108, full
    for i = 1:2000-1;
        view(ml3117, i, i+1:2000)[:] = view(ml3117, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3120, ml3108, 1.0, ml3117)

    # tmp32: ml3118, full, tmp31: ml3117, full
    ml3121 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3117, ml3121, info) = LinearAlgebra.LAPACK.getrf!(ml3117)

    # tmp32: ml3118, full, P35: ml3121, ipiv, L33: ml3117, lower_triangular_udiag, U34: ml3117, upper_triangular
    ml3122 = [1:length(ml3121);]
    @inbounds for i in 1:length(ml3121)
        ml3122[i], ml3122[ml3121[i]] = ml3122[ml3121[i]], ml3122[i];
    end;
    ml3123 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3123 = ml3118[ml3122]

    # L33: ml3117, lower_triangular_udiag, U34: ml3117, upper_triangular, tmp40: ml3123, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3117, ml3123)

    # U34: ml3117, upper_triangular, tmp41: ml3123, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3117, ml3123)

    # tmp17: ml3123, full
    # x = tmp17
    return (ml3123)
end