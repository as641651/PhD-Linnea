using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm85(ml2842::Array{Float64,2}, ml2843::Array{Float64,2}, ml2844::Array{Float64,2}, ml2845::Array{Float64,2}, ml2846::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2842, full, L: ml2843, full, A: ml2844, full, B: ml2845, full, y: ml2846, full
    ml2847 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2847, ml2845)

    # R: ml2842, full, L: ml2843, full, A: ml2844, full, y: ml2846, full, tmp26: ml2847, full
    ml2848 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2844, ml2848, info) = LinearAlgebra.LAPACK.getrf!(ml2844)

    # R: ml2842, full, L: ml2843, full, y: ml2846, full, tmp26: ml2847, full, P11: ml2848, ipiv, L9: ml2844, lower_triangular_udiag, U10: ml2844, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2844, ml2847)

    # R: ml2842, full, L: ml2843, full, y: ml2846, full, P11: ml2848, ipiv, L9: ml2844, lower_triangular_udiag, tmp27: ml2847, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2844, ml2847)

    # R: ml2842, full, L: ml2843, full, y: ml2846, full, P11: ml2848, ipiv, tmp28: ml2847, full
    ml2849 = [1:length(ml2848);]
    @inbounds for i in 1:length(ml2848)
        ml2849[i], ml2849[ml2848[i]] = ml2849[ml2848[i]], ml2849[i];
    end;
    ml2850 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2850 = ml2847[invperm(ml2849),:]

    # R: ml2842, full, L: ml2843, full, y: ml2846, full, tmp25: ml2850, full
    ml2851 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2850, 0.0, ml2851)

    # R: ml2842, full, L: ml2843, full, y: ml2846, full, tmp19: ml2851, symmetric_lower_triangular
    ml2852 = diag(ml2843)
    ml2853 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2842, 1, ml2853, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2842, 2);
        view(ml2842, :, i)[:] .*= ml2852;
    end;        

    # R: ml2853, full, y: ml2846, full, tmp19: ml2851, symmetric_lower_triangular, tmp29: ml2842, full
    for i = 1:2000-1;
        view(ml2851, i, i+1:2000)[:] = view(ml2851, i+1:2000, i);
    end;
    ml2854 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2851, 1, ml2854, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2842, ml2853, 1.0, ml2851)

    # y: ml2846, full, tmp19: ml2854, full, tmp31: ml2851, full
    ml2855 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2851, ml2855, info) = LinearAlgebra.LAPACK.getrf!(ml2851)

    # y: ml2846, full, tmp19: ml2854, full, P35: ml2855, ipiv, L33: ml2851, lower_triangular_udiag, U34: ml2851, upper_triangular
    ml2856 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2854, ml2846, 0.0, ml2856)

    # P35: ml2855, ipiv, L33: ml2851, lower_triangular_udiag, U34: ml2851, upper_triangular, tmp32: ml2856, full
    ml2857 = [1:length(ml2855);]
    @inbounds for i in 1:length(ml2855)
        ml2857[i], ml2857[ml2855[i]] = ml2857[ml2855[i]], ml2857[i];
    end;
    ml2858 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2858 = ml2856[ml2857]

    # L33: ml2851, lower_triangular_udiag, U34: ml2851, upper_triangular, tmp40: ml2858, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2851, ml2858)

    # U34: ml2851, upper_triangular, tmp41: ml2858, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2851, ml2858)

    # tmp17: ml2858, full
    # x = tmp17
    return (ml2858)
end