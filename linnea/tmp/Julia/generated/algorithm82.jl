using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm82(ml2744::Array{Float64,2}, ml2745::Array{Float64,2}, ml2746::Array{Float64,2}, ml2747::Array{Float64,2}, ml2748::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2744, full, L: ml2745, full, A: ml2746, full, B: ml2747, full, y: ml2748, full
    ml2749 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2749, ml2747)

    # R: ml2744, full, L: ml2745, full, A: ml2746, full, y: ml2748, full, tmp26: ml2749, full
    ml2750 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2746, ml2750, info) = LinearAlgebra.LAPACK.getrf!(ml2746)

    # R: ml2744, full, L: ml2745, full, y: ml2748, full, tmp26: ml2749, full, P11: ml2750, ipiv, L9: ml2746, lower_triangular_udiag, U10: ml2746, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2746, ml2749)

    # R: ml2744, full, L: ml2745, full, y: ml2748, full, P11: ml2750, ipiv, L9: ml2746, lower_triangular_udiag, tmp27: ml2749, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2746, ml2749)

    # R: ml2744, full, L: ml2745, full, y: ml2748, full, P11: ml2750, ipiv, tmp28: ml2749, full
    ml2751 = [1:length(ml2750);]
    @inbounds for i in 1:length(ml2750)
        ml2751[i], ml2751[ml2750[i]] = ml2751[ml2750[i]], ml2751[i];
    end;
    ml2752 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2752 = ml2749[invperm(ml2751),:]

    # R: ml2744, full, L: ml2745, full, y: ml2748, full, tmp25: ml2752, full
    ml2753 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2752, 0.0, ml2753)

    # R: ml2744, full, L: ml2745, full, y: ml2748, full, tmp19: ml2753, symmetric_lower_triangular
    ml2754 = diag(ml2745)
    ml2755 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2744, 1, ml2755, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2744, 2);
        view(ml2744, :, i)[:] .*= ml2754;
    end;        

    # R: ml2755, full, y: ml2748, full, tmp19: ml2753, symmetric_lower_triangular, tmp29: ml2744, full
    ml2756 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2753, ml2748, 0.0, ml2756)

    # R: ml2755, full, tmp19: ml2753, symmetric_lower_triangular, tmp29: ml2744, full, tmp32: ml2756, full
    for i = 1:2000-1;
        view(ml2753, i, i+1:2000)[:] = view(ml2753, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2744, ml2755, 1.0, ml2753)

    # tmp32: ml2756, full, tmp31: ml2753, full
    ml2757 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2753, ml2757, info) = LinearAlgebra.LAPACK.getrf!(ml2753)

    # tmp32: ml2756, full, P35: ml2757, ipiv, L33: ml2753, lower_triangular_udiag, U34: ml2753, upper_triangular
    ml2758 = [1:length(ml2757);]
    @inbounds for i in 1:length(ml2757)
        ml2758[i], ml2758[ml2757[i]] = ml2758[ml2757[i]], ml2758[i];
    end;
    ml2759 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2759 = ml2756[ml2758]

    # L33: ml2753, lower_triangular_udiag, U34: ml2753, upper_triangular, tmp40: ml2759, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2753, ml2759)

    # U34: ml2753, upper_triangular, tmp41: ml2759, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2753, ml2759)

    # tmp17: ml2759, full
    # x = tmp17
    return (ml2759)
end