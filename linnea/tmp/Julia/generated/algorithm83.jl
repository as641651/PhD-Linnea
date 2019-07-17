using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm83(ml2776::Array{Float64,2}, ml2777::Array{Float64,2}, ml2778::Array{Float64,2}, ml2779::Array{Float64,2}, ml2780::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2776, full, L: ml2777, full, A: ml2778, full, B: ml2779, full, y: ml2780, full
    ml2781 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2781, ml2779)

    # R: ml2776, full, L: ml2777, full, A: ml2778, full, y: ml2780, full, tmp26: ml2781, full
    ml2782 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2778, ml2782, info) = LinearAlgebra.LAPACK.getrf!(ml2778)

    # R: ml2776, full, L: ml2777, full, y: ml2780, full, tmp26: ml2781, full, P11: ml2782, ipiv, L9: ml2778, lower_triangular_udiag, U10: ml2778, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2778, ml2781)

    # R: ml2776, full, L: ml2777, full, y: ml2780, full, P11: ml2782, ipiv, L9: ml2778, lower_triangular_udiag, tmp27: ml2781, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2778, ml2781)

    # R: ml2776, full, L: ml2777, full, y: ml2780, full, P11: ml2782, ipiv, tmp28: ml2781, full
    ml2783 = [1:length(ml2782);]
    @inbounds for i in 1:length(ml2782)
        ml2783[i], ml2783[ml2782[i]] = ml2783[ml2782[i]], ml2783[i];
    end;
    ml2784 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2784 = ml2781[invperm(ml2783),:]

    # R: ml2776, full, L: ml2777, full, y: ml2780, full, tmp25: ml2784, full
    ml2785 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2784, 0.0, ml2785)

    # R: ml2776, full, L: ml2777, full, y: ml2780, full, tmp19: ml2785, symmetric_lower_triangular
    ml2786 = diag(ml2777)
    ml2787 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2776, 1, ml2787, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2776, 2);
        view(ml2776, :, i)[:] .*= ml2786;
    end;        

    # R: ml2787, full, y: ml2780, full, tmp19: ml2785, symmetric_lower_triangular, tmp29: ml2776, full
    ml2788 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2785, ml2780, 0.0, ml2788)

    # R: ml2787, full, tmp19: ml2785, symmetric_lower_triangular, tmp29: ml2776, full, tmp32: ml2788, full
    for i = 1:2000-1;
        view(ml2785, i, i+1:2000)[:] = view(ml2785, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2776, ml2787, 1.0, ml2785)

    # tmp32: ml2788, full, tmp31: ml2785, full
    ml2789 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2785, ml2789, info) = LinearAlgebra.LAPACK.getrf!(ml2785)

    # tmp32: ml2788, full, P35: ml2789, ipiv, L33: ml2785, lower_triangular_udiag, U34: ml2785, upper_triangular
    ml2790 = [1:length(ml2789);]
    @inbounds for i in 1:length(ml2789)
        ml2790[i], ml2790[ml2789[i]] = ml2790[ml2789[i]], ml2790[i];
    end;
    ml2791 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2791 = ml2788[ml2790]

    # L33: ml2785, lower_triangular_udiag, U34: ml2785, upper_triangular, tmp40: ml2791, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2785, ml2791)

    # U34: ml2785, upper_triangular, tmp41: ml2791, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2785, ml2791)

    # tmp17: ml2791, full
    # x = tmp17
    return (ml2791)
end