using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm89(ml2978::Array{Float64,2}, ml2979::Array{Float64,2}, ml2980::Array{Float64,2}, ml2981::Array{Float64,2}, ml2982::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2978, full, L: ml2979, full, A: ml2980, full, B: ml2981, full, y: ml2982, full
    ml2983 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2980, ml2983, info) = LinearAlgebra.LAPACK.getrf!(ml2980)

    # R: ml2978, full, L: ml2979, full, B: ml2981, full, y: ml2982, full, P11: ml2983, ipiv, L9: ml2980, lower_triangular_udiag, U10: ml2980, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml2980, ml2981)

    # R: ml2978, full, L: ml2979, full, y: ml2982, full, P11: ml2983, ipiv, L9: ml2980, lower_triangular_udiag, tmp53: ml2981, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml2980, ml2981)

    # R: ml2978, full, L: ml2979, full, y: ml2982, full, P11: ml2983, ipiv, tmp54: ml2981, full
    ml2984 = [1:length(ml2983);]
    @inbounds for i in 1:length(ml2983)
        ml2984[i], ml2984[ml2983[i]] = ml2984[ml2983[i]], ml2984[i];
    end;
    ml2985 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml2985 = ml2981[:,invperm(ml2984)]

    # R: ml2978, full, L: ml2979, full, y: ml2982, full, tmp55: ml2985, full
    ml2986 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml2986, ml2985)

    # R: ml2978, full, L: ml2979, full, y: ml2982, full, tmp25: ml2986, full
    ml2987 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2986, 0.0, ml2987)

    # R: ml2978, full, L: ml2979, full, y: ml2982, full, tmp19: ml2987, symmetric_lower_triangular
    ml2988 = diag(ml2979)
    ml2989 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2978, 1, ml2989, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2978, 2);
        view(ml2978, :, i)[:] .*= ml2988;
    end;        

    # R: ml2989, full, y: ml2982, full, tmp19: ml2987, symmetric_lower_triangular, tmp29: ml2978, full
    for i = 1:2000-1;
        view(ml2987, i, i+1:2000)[:] = view(ml2987, i+1:2000, i);
    end;
    ml2990 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2987, 1, ml2990, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2978, ml2989, 1.0, ml2987)

    # y: ml2982, full, tmp19: ml2990, full, tmp31: ml2987, full
    ml2991 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2987, ml2991, info) = LinearAlgebra.LAPACK.getrf!(ml2987)

    # y: ml2982, full, tmp19: ml2990, full, P35: ml2991, ipiv, L33: ml2987, lower_triangular_udiag, U34: ml2987, upper_triangular
    ml2992 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2990, ml2982, 0.0, ml2992)

    # P35: ml2991, ipiv, L33: ml2987, lower_triangular_udiag, U34: ml2987, upper_triangular, tmp32: ml2992, full
    ml2993 = [1:length(ml2991);]
    @inbounds for i in 1:length(ml2991)
        ml2993[i], ml2993[ml2991[i]] = ml2993[ml2991[i]], ml2993[i];
    end;
    ml2994 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2994 = ml2992[ml2993]

    # L33: ml2987, lower_triangular_udiag, U34: ml2987, upper_triangular, tmp40: ml2994, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2987, ml2994)

    # U34: ml2987, upper_triangular, tmp41: ml2994, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2987, ml2994)

    # tmp17: ml2994, full
    # x = tmp17
    return (ml2994)
end