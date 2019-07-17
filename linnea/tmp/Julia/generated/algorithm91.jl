using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm91(ml3044::Array{Float64,2}, ml3045::Array{Float64,2}, ml3046::Array{Float64,2}, ml3047::Array{Float64,2}, ml3048::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3044, full, L: ml3045, full, A: ml3046, full, B: ml3047, full, y: ml3048, full
    ml3049 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3049, ml3047)

    # R: ml3044, full, L: ml3045, full, A: ml3046, full, y: ml3048, full, tmp26: ml3049, full
    ml3050 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3046, ml3050, info) = LinearAlgebra.LAPACK.getrf!(ml3046)

    # R: ml3044, full, L: ml3045, full, y: ml3048, full, tmp26: ml3049, full, P11: ml3050, ipiv, L9: ml3046, lower_triangular_udiag, U10: ml3046, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3046, ml3049)

    # R: ml3044, full, L: ml3045, full, y: ml3048, full, P11: ml3050, ipiv, L9: ml3046, lower_triangular_udiag, tmp27: ml3049, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3046, ml3049)

    # R: ml3044, full, L: ml3045, full, y: ml3048, full, P11: ml3050, ipiv, tmp28: ml3049, full
    ml3051 = [1:length(ml3050);]
    @inbounds for i in 1:length(ml3050)
        ml3051[i], ml3051[ml3050[i]] = ml3051[ml3050[i]], ml3051[i];
    end;
    ml3052 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3052 = ml3049[invperm(ml3051),:]

    # R: ml3044, full, L: ml3045, full, y: ml3048, full, tmp25: ml3052, full
    ml3053 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3052, 0.0, ml3053)

    # R: ml3044, full, L: ml3045, full, y: ml3048, full, tmp19: ml3053, symmetric_lower_triangular
    ml3054 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3053, ml3048, 0.0, ml3054)

    # R: ml3044, full, L: ml3045, full, tmp19: ml3053, symmetric_lower_triangular, tmp32: ml3054, full
    ml3055 = diag(ml3045)
    ml3056 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3044, 1, ml3056, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3044, 2);
        view(ml3044, :, i)[:] .*= ml3055;
    end;        

    # R: ml3056, full, tmp19: ml3053, symmetric_lower_triangular, tmp32: ml3054, full, tmp29: ml3044, full
    for i = 1:2000-1;
        view(ml3053, i, i+1:2000)[:] = view(ml3053, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3056, ml3044, 1.0, ml3053)

    # tmp32: ml3054, full, tmp31: ml3053, full
    ml3057 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3053, ml3057, info) = LinearAlgebra.LAPACK.getrf!(ml3053)

    # tmp32: ml3054, full, P35: ml3057, ipiv, L33: ml3053, lower_triangular_udiag, U34: ml3053, upper_triangular
    ml3058 = [1:length(ml3057);]
    @inbounds for i in 1:length(ml3057)
        ml3058[i], ml3058[ml3057[i]] = ml3058[ml3057[i]], ml3058[i];
    end;
    ml3059 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3059 = ml3054[ml3058]

    # L33: ml3053, lower_triangular_udiag, U34: ml3053, upper_triangular, tmp40: ml3059, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3053, ml3059)

    # U34: ml3053, upper_triangular, tmp41: ml3059, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3053, ml3059)

    # tmp17: ml3059, full
    # x = tmp17
    return (ml3059)
end