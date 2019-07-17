using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm84(ml2808::Array{Float64,2}, ml2809::Array{Float64,2}, ml2810::Array{Float64,2}, ml2811::Array{Float64,2}, ml2812::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2808, full, L: ml2809, full, A: ml2810, full, B: ml2811, full, y: ml2812, full
    ml2813 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2813, ml2811)

    # R: ml2808, full, L: ml2809, full, A: ml2810, full, y: ml2812, full, tmp26: ml2813, full
    ml2814 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2810, ml2814, info) = LinearAlgebra.LAPACK.getrf!(ml2810)

    # R: ml2808, full, L: ml2809, full, y: ml2812, full, tmp26: ml2813, full, P11: ml2814, ipiv, L9: ml2810, lower_triangular_udiag, U10: ml2810, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2810, ml2813)

    # R: ml2808, full, L: ml2809, full, y: ml2812, full, P11: ml2814, ipiv, L9: ml2810, lower_triangular_udiag, tmp27: ml2813, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2810, ml2813)

    # R: ml2808, full, L: ml2809, full, y: ml2812, full, P11: ml2814, ipiv, tmp28: ml2813, full
    ml2815 = [1:length(ml2814);]
    @inbounds for i in 1:length(ml2814)
        ml2815[i], ml2815[ml2814[i]] = ml2815[ml2814[i]], ml2815[i];
    end;
    ml2816 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2816 = ml2813[invperm(ml2815),:]

    # R: ml2808, full, L: ml2809, full, y: ml2812, full, tmp25: ml2816, full
    ml2817 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2816, 0.0, ml2817)

    # R: ml2808, full, L: ml2809, full, y: ml2812, full, tmp19: ml2817, symmetric_lower_triangular
    ml2818 = diag(ml2809)
    ml2819 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2808, 1, ml2819, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2808, 2);
        view(ml2808, :, i)[:] .*= ml2818;
    end;        

    # R: ml2819, full, y: ml2812, full, tmp19: ml2817, symmetric_lower_triangular, tmp29: ml2808, full
    for i = 1:2000-1;
        view(ml2817, i, i+1:2000)[:] = view(ml2817, i+1:2000, i);
    end;
    ml2820 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2817, 1, ml2820, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2808, ml2819, 1.0, ml2817)

    # y: ml2812, full, tmp19: ml2820, full, tmp31: ml2817, full
    ml2821 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2817, ml2821, info) = LinearAlgebra.LAPACK.getrf!(ml2817)

    # y: ml2812, full, tmp19: ml2820, full, P35: ml2821, ipiv, L33: ml2817, lower_triangular_udiag, U34: ml2817, upper_triangular
    ml2822 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2820, ml2812, 0.0, ml2822)

    # P35: ml2821, ipiv, L33: ml2817, lower_triangular_udiag, U34: ml2817, upper_triangular, tmp32: ml2822, full
    ml2823 = [1:length(ml2821);]
    @inbounds for i in 1:length(ml2821)
        ml2823[i], ml2823[ml2821[i]] = ml2823[ml2821[i]], ml2823[i];
    end;
    ml2824 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2824 = ml2822[ml2823]

    # L33: ml2817, lower_triangular_udiag, U34: ml2817, upper_triangular, tmp40: ml2824, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2817, ml2824)

    # U34: ml2817, upper_triangular, tmp41: ml2824, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2817, ml2824)

    # tmp17: ml2824, full
    # x = tmp17
    return (ml2824)
end