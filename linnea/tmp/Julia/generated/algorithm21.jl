using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm21(ml704::Array{Float64,2}, ml705::Array{Float64,2}, ml706::Array{Float64,2}, ml707::Array{Float64,2}, ml708::Array{Float64,1})
    # cost 5.07e+10
    # R: ml704, full, L: ml705, full, A: ml706, full, B: ml707, full, y: ml708, full
    ml709 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml706, ml709, info) = LinearAlgebra.LAPACK.getrf!(ml706)

    # R: ml704, full, L: ml705, full, B: ml707, full, y: ml708, full, P11: ml709, ipiv, L9: ml706, lower_triangular_udiag, U10: ml706, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml706, ml707)

    # R: ml704, full, L: ml705, full, y: ml708, full, P11: ml709, ipiv, L9: ml706, lower_triangular_udiag, tmp53: ml707, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml706, ml707)

    # R: ml704, full, L: ml705, full, y: ml708, full, P11: ml709, ipiv, tmp54: ml707, full
    ml710 = [1:length(ml709);]
    @inbounds for i in 1:length(ml709)
        ml710[i], ml710[ml709[i]] = ml710[ml709[i]], ml710[i];
    end;
    ml711 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml711 = ml707[:,invperm(ml710)]

    # R: ml704, full, L: ml705, full, y: ml708, full, tmp55: ml711, full
    ml712 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml712, ml711)

    # R: ml704, full, L: ml705, full, y: ml708, full, tmp25: ml712, full
    ml713 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml712, 0.0, ml713)

    # R: ml704, full, L: ml705, full, y: ml708, full, tmp19: ml713, symmetric_lower_triangular
    ml714 = diag(ml705)
    ml715 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml704, 1, ml715, 1)
    # tmp29 = (L R)
    for i = 1:size(ml704, 2);
        view(ml704, :, i)[:] .*= ml714;
    end;        

    # R: ml715, full, y: ml708, full, tmp19: ml713, symmetric_lower_triangular, tmp29: ml704, full
    ml716 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml713, ml708, 0.0, ml716)

    # R: ml715, full, tmp19: ml713, symmetric_lower_triangular, tmp29: ml704, full, tmp32: ml716, full
    for i = 1:2000-1;
        view(ml713, i, i+1:2000)[:] = view(ml713, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml704, ml715, 1.0, ml713)

    # tmp32: ml716, full, tmp31: ml713, full
    ml717 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml713, ml717, info) = LinearAlgebra.LAPACK.getrf!(ml713)

    # tmp32: ml716, full, P35: ml717, ipiv, L33: ml713, lower_triangular_udiag, U34: ml713, upper_triangular
    ml718 = [1:length(ml717);]
    @inbounds for i in 1:length(ml717)
        ml718[i], ml718[ml717[i]] = ml718[ml717[i]], ml718[i];
    end;
    ml719 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml719 = ml716[ml718]

    # L33: ml713, lower_triangular_udiag, U34: ml713, upper_triangular, tmp40: ml719, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml713, ml719)

    # U34: ml713, upper_triangular, tmp41: ml719, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml713, ml719)

    # tmp17: ml719, full
    # x = tmp17
    return (ml719)
end