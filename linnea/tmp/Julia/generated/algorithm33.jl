using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm33(ml1100::Array{Float64,2}, ml1101::Array{Float64,2}, ml1102::Array{Float64,2}, ml1103::Array{Float64,2}, ml1104::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1100, full, L: ml1101, full, A: ml1102, full, B: ml1103, full, y: ml1104, full
    ml1105 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1105, ml1103)

    # R: ml1100, full, L: ml1101, full, A: ml1102, full, y: ml1104, full, tmp26: ml1105, full
    ml1106 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1102, ml1106, info) = LinearAlgebra.LAPACK.getrf!(ml1102)

    # R: ml1100, full, L: ml1101, full, y: ml1104, full, tmp26: ml1105, full, P11: ml1106, ipiv, L9: ml1102, lower_triangular_udiag, U10: ml1102, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1102, ml1105)

    # R: ml1100, full, L: ml1101, full, y: ml1104, full, P11: ml1106, ipiv, L9: ml1102, lower_triangular_udiag, tmp27: ml1105, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1102, ml1105)

    # R: ml1100, full, L: ml1101, full, y: ml1104, full, P11: ml1106, ipiv, tmp28: ml1105, full
    ml1107 = [1:length(ml1106);]
    @inbounds for i in 1:length(ml1106)
        ml1107[i], ml1107[ml1106[i]] = ml1107[ml1106[i]], ml1107[i];
    end;
    ml1108 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1108 = ml1105[invperm(ml1107),:]

    # R: ml1100, full, L: ml1101, full, y: ml1104, full, tmp25: ml1108, full
    ml1109 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1108, 0.0, ml1109)

    # R: ml1100, full, L: ml1101, full, y: ml1104, full, tmp19: ml1109, symmetric_lower_triangular
    ml1110 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1109, ml1104, 0.0, ml1110)

    # R: ml1100, full, L: ml1101, full, tmp19: ml1109, symmetric_lower_triangular, tmp32: ml1110, full
    ml1111 = diag(ml1101)
    ml1112 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1100, 1, ml1112, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1100, 2);
        view(ml1100, :, i)[:] .*= ml1111;
    end;        

    # R: ml1112, full, tmp19: ml1109, symmetric_lower_triangular, tmp32: ml1110, full, tmp29: ml1100, full
    for i = 1:2000-1;
        view(ml1109, i, i+1:2000)[:] = view(ml1109, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1112, ml1100, 1.0, ml1109)

    # tmp32: ml1110, full, tmp31: ml1109, full
    ml1113 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1109, ml1113, info) = LinearAlgebra.LAPACK.getrf!(ml1109)

    # tmp32: ml1110, full, P35: ml1113, ipiv, L33: ml1109, lower_triangular_udiag, U34: ml1109, upper_triangular
    ml1114 = [1:length(ml1113);]
    @inbounds for i in 1:length(ml1113)
        ml1114[i], ml1114[ml1113[i]] = ml1114[ml1113[i]], ml1114[i];
    end;
    ml1115 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1115 = ml1110[ml1114]

    # L33: ml1109, lower_triangular_udiag, U34: ml1109, upper_triangular, tmp40: ml1115, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1109, ml1115)

    # U34: ml1109, upper_triangular, tmp41: ml1115, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1109, ml1115)

    # tmp17: ml1115, full
    # x = tmp17
    return (ml1115)
end