using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm99(ml3310::Array{Float64,2}, ml3311::Array{Float64,2}, ml3312::Array{Float64,2}, ml3313::Array{Float64,2}, ml3314::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3310, full, L: ml3311, full, A: ml3312, full, B: ml3313, full, y: ml3314, full
    ml3315 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3315, ml3313)

    # R: ml3310, full, L: ml3311, full, A: ml3312, full, y: ml3314, full, tmp26: ml3315, full
    ml3316 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3312, ml3316, info) = LinearAlgebra.LAPACK.getrf!(ml3312)

    # R: ml3310, full, L: ml3311, full, y: ml3314, full, tmp26: ml3315, full, P11: ml3316, ipiv, L9: ml3312, lower_triangular_udiag, U10: ml3312, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3312, ml3315)

    # R: ml3310, full, L: ml3311, full, y: ml3314, full, P11: ml3316, ipiv, L9: ml3312, lower_triangular_udiag, tmp27: ml3315, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3312, ml3315)

    # R: ml3310, full, L: ml3311, full, y: ml3314, full, P11: ml3316, ipiv, tmp28: ml3315, full
    ml3317 = [1:length(ml3316);]
    @inbounds for i in 1:length(ml3316)
        ml3317[i], ml3317[ml3316[i]] = ml3317[ml3316[i]], ml3317[i];
    end;
    ml3318 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3318 = ml3315[invperm(ml3317),:]

    # R: ml3310, full, L: ml3311, full, y: ml3314, full, tmp25: ml3318, full
    ml3319 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3318, 0.0, ml3319)

    # R: ml3310, full, L: ml3311, full, y: ml3314, full, tmp19: ml3319, symmetric_lower_triangular
    ml3320 = diag(ml3311)
    ml3321 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3310, 1, ml3321, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3310, 2);
        view(ml3310, :, i)[:] .*= ml3320;
    end;        

    # R: ml3321, full, y: ml3314, full, tmp19: ml3319, symmetric_lower_triangular, tmp29: ml3310, full
    for i = 1:2000-1;
        view(ml3319, i, i+1:2000)[:] = view(ml3319, i+1:2000, i);
    end;
    ml3322 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3319, 1, ml3322, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3310, ml3321, 1.0, ml3319)

    # y: ml3314, full, tmp19: ml3322, full, tmp31: ml3319, full
    ml3323 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3319, ml3323, info) = LinearAlgebra.LAPACK.getrf!(ml3319)

    # y: ml3314, full, tmp19: ml3322, full, P35: ml3323, ipiv, L33: ml3319, lower_triangular_udiag, U34: ml3319, upper_triangular
    ml3324 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3322, ml3314, 0.0, ml3324)

    # P35: ml3323, ipiv, L33: ml3319, lower_triangular_udiag, U34: ml3319, upper_triangular, tmp32: ml3324, full
    ml3325 = [1:length(ml3323);]
    @inbounds for i in 1:length(ml3323)
        ml3325[i], ml3325[ml3323[i]] = ml3325[ml3323[i]], ml3325[i];
    end;
    ml3326 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3326 = ml3324[ml3325]

    # L33: ml3319, lower_triangular_udiag, U34: ml3319, upper_triangular, tmp40: ml3326, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3319, ml3326)

    # U34: ml3319, upper_triangular, tmp41: ml3326, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3319, ml3326)

    # tmp17: ml3326, full
    # x = tmp17
    return (ml3326)
end