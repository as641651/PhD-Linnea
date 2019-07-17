using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm87(ml2910::Array{Float64,2}, ml2911::Array{Float64,2}, ml2912::Array{Float64,2}, ml2913::Array{Float64,2}, ml2914::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2910, full, L: ml2911, full, A: ml2912, full, B: ml2913, full, y: ml2914, full
    ml2915 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2915, ml2913)

    # R: ml2910, full, L: ml2911, full, A: ml2912, full, y: ml2914, full, tmp26: ml2915, full
    ml2916 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2912, ml2916, info) = LinearAlgebra.LAPACK.getrf!(ml2912)

    # R: ml2910, full, L: ml2911, full, y: ml2914, full, tmp26: ml2915, full, P11: ml2916, ipiv, L9: ml2912, lower_triangular_udiag, U10: ml2912, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2912, ml2915)

    # R: ml2910, full, L: ml2911, full, y: ml2914, full, P11: ml2916, ipiv, L9: ml2912, lower_triangular_udiag, tmp27: ml2915, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2912, ml2915)

    # R: ml2910, full, L: ml2911, full, y: ml2914, full, P11: ml2916, ipiv, tmp28: ml2915, full
    ml2917 = [1:length(ml2916);]
    @inbounds for i in 1:length(ml2916)
        ml2917[i], ml2917[ml2916[i]] = ml2917[ml2916[i]], ml2917[i];
    end;
    ml2918 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2918 = ml2915[invperm(ml2917),:]

    # R: ml2910, full, L: ml2911, full, y: ml2914, full, tmp25: ml2918, full
    ml2919 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2918, 0.0, ml2919)

    # R: ml2910, full, L: ml2911, full, y: ml2914, full, tmp19: ml2919, symmetric_lower_triangular
    ml2920 = diag(ml2911)
    ml2921 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2910, 1, ml2921, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2910, 2);
        view(ml2910, :, i)[:] .*= ml2920;
    end;        

    # R: ml2921, full, y: ml2914, full, tmp19: ml2919, symmetric_lower_triangular, tmp29: ml2910, full
    for i = 1:2000-1;
        view(ml2919, i, i+1:2000)[:] = view(ml2919, i+1:2000, i);
    end;
    ml2922 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2919, 1, ml2922, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2910, ml2921, 1.0, ml2919)

    # y: ml2914, full, tmp19: ml2922, full, tmp31: ml2919, full
    ml2923 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2919, ml2923, info) = LinearAlgebra.LAPACK.getrf!(ml2919)

    # y: ml2914, full, tmp19: ml2922, full, P35: ml2923, ipiv, L33: ml2919, lower_triangular_udiag, U34: ml2919, upper_triangular
    ml2924 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2922, ml2914, 0.0, ml2924)

    # P35: ml2923, ipiv, L33: ml2919, lower_triangular_udiag, U34: ml2919, upper_triangular, tmp32: ml2924, full
    ml2925 = [1:length(ml2923);]
    @inbounds for i in 1:length(ml2923)
        ml2925[i], ml2925[ml2923[i]] = ml2925[ml2923[i]], ml2925[i];
    end;
    ml2926 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2926 = ml2924[ml2925]

    # L33: ml2919, lower_triangular_udiag, U34: ml2919, upper_triangular, tmp40: ml2926, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2919, ml2926)

    # U34: ml2919, upper_triangular, tmp41: ml2926, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2919, ml2926)

    # tmp17: ml2926, full
    # x = tmp17
    return (ml2926)
end