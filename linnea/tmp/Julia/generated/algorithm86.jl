using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm86(ml2876::Array{Float64,2}, ml2877::Array{Float64,2}, ml2878::Array{Float64,2}, ml2879::Array{Float64,2}, ml2880::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2876, full, L: ml2877, full, A: ml2878, full, B: ml2879, full, y: ml2880, full
    ml2881 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2881, ml2879)

    # R: ml2876, full, L: ml2877, full, A: ml2878, full, y: ml2880, full, tmp26: ml2881, full
    ml2882 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2878, ml2882, info) = LinearAlgebra.LAPACK.getrf!(ml2878)

    # R: ml2876, full, L: ml2877, full, y: ml2880, full, tmp26: ml2881, full, P11: ml2882, ipiv, L9: ml2878, lower_triangular_udiag, U10: ml2878, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2878, ml2881)

    # R: ml2876, full, L: ml2877, full, y: ml2880, full, P11: ml2882, ipiv, L9: ml2878, lower_triangular_udiag, tmp27: ml2881, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2878, ml2881)

    # R: ml2876, full, L: ml2877, full, y: ml2880, full, P11: ml2882, ipiv, tmp28: ml2881, full
    ml2883 = [1:length(ml2882);]
    @inbounds for i in 1:length(ml2882)
        ml2883[i], ml2883[ml2882[i]] = ml2883[ml2882[i]], ml2883[i];
    end;
    ml2884 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2884 = ml2881[invperm(ml2883),:]

    # R: ml2876, full, L: ml2877, full, y: ml2880, full, tmp25: ml2884, full
    ml2885 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2884, 0.0, ml2885)

    # R: ml2876, full, L: ml2877, full, y: ml2880, full, tmp19: ml2885, symmetric_lower_triangular
    ml2886 = diag(ml2877)
    ml2887 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2876, 1, ml2887, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2876, 2);
        view(ml2876, :, i)[:] .*= ml2886;
    end;        

    # R: ml2887, full, y: ml2880, full, tmp19: ml2885, symmetric_lower_triangular, tmp29: ml2876, full
    for i = 1:2000-1;
        view(ml2885, i, i+1:2000)[:] = view(ml2885, i+1:2000, i);
    end;
    ml2888 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2885, 1, ml2888, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2876, ml2887, 1.0, ml2885)

    # y: ml2880, full, tmp19: ml2888, full, tmp31: ml2885, full
    ml2889 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2885, ml2889, info) = LinearAlgebra.LAPACK.getrf!(ml2885)

    # y: ml2880, full, tmp19: ml2888, full, P35: ml2889, ipiv, L33: ml2885, lower_triangular_udiag, U34: ml2885, upper_triangular
    ml2890 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2888, ml2880, 0.0, ml2890)

    # P35: ml2889, ipiv, L33: ml2885, lower_triangular_udiag, U34: ml2885, upper_triangular, tmp32: ml2890, full
    ml2891 = [1:length(ml2889);]
    @inbounds for i in 1:length(ml2889)
        ml2891[i], ml2891[ml2889[i]] = ml2891[ml2889[i]], ml2891[i];
    end;
    ml2892 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2892 = ml2890[ml2891]

    # L33: ml2885, lower_triangular_udiag, U34: ml2885, upper_triangular, tmp40: ml2892, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2885, ml2892)

    # U34: ml2885, upper_triangular, tmp41: ml2892, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2885, ml2892)

    # tmp17: ml2892, full
    # x = tmp17
    return (ml2892)
end