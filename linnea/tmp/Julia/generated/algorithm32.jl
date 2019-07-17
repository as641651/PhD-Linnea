using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm32(ml1068::Array{Float64,2}, ml1069::Array{Float64,2}, ml1070::Array{Float64,2}, ml1071::Array{Float64,2}, ml1072::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1068, full, L: ml1069, full, A: ml1070, full, B: ml1071, full, y: ml1072, full
    ml1073 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1073, ml1071)

    # R: ml1068, full, L: ml1069, full, A: ml1070, full, y: ml1072, full, tmp26: ml1073, full
    ml1074 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1070, ml1074, info) = LinearAlgebra.LAPACK.getrf!(ml1070)

    # R: ml1068, full, L: ml1069, full, y: ml1072, full, tmp26: ml1073, full, P11: ml1074, ipiv, L9: ml1070, lower_triangular_udiag, U10: ml1070, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1070, ml1073)

    # R: ml1068, full, L: ml1069, full, y: ml1072, full, P11: ml1074, ipiv, L9: ml1070, lower_triangular_udiag, tmp27: ml1073, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1070, ml1073)

    # R: ml1068, full, L: ml1069, full, y: ml1072, full, P11: ml1074, ipiv, tmp28: ml1073, full
    ml1075 = [1:length(ml1074);]
    @inbounds for i in 1:length(ml1074)
        ml1075[i], ml1075[ml1074[i]] = ml1075[ml1074[i]], ml1075[i];
    end;
    ml1076 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1076 = ml1073[invperm(ml1075),:]

    # R: ml1068, full, L: ml1069, full, y: ml1072, full, tmp25: ml1076, full
    ml1077 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1076, 0.0, ml1077)

    # R: ml1068, full, L: ml1069, full, y: ml1072, full, tmp19: ml1077, symmetric_lower_triangular
    ml1078 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1077, ml1072, 0.0, ml1078)

    # R: ml1068, full, L: ml1069, full, tmp19: ml1077, symmetric_lower_triangular, tmp32: ml1078, full
    ml1079 = diag(ml1069)
    ml1080 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1068, 1, ml1080, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1068, 2);
        view(ml1068, :, i)[:] .*= ml1079;
    end;        

    # R: ml1080, full, tmp19: ml1077, symmetric_lower_triangular, tmp32: ml1078, full, tmp29: ml1068, full
    for i = 1:2000-1;
        view(ml1077, i, i+1:2000)[:] = view(ml1077, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1080, ml1068, 1.0, ml1077)

    # tmp32: ml1078, full, tmp31: ml1077, full
    ml1081 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1077, ml1081, info) = LinearAlgebra.LAPACK.getrf!(ml1077)

    # tmp32: ml1078, full, P35: ml1081, ipiv, L33: ml1077, lower_triangular_udiag, U34: ml1077, upper_triangular
    ml1082 = [1:length(ml1081);]
    @inbounds for i in 1:length(ml1081)
        ml1082[i], ml1082[ml1081[i]] = ml1082[ml1081[i]], ml1082[i];
    end;
    ml1083 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1083 = ml1078[ml1082]

    # L33: ml1077, lower_triangular_udiag, U34: ml1077, upper_triangular, tmp40: ml1083, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1077, ml1083)

    # U34: ml1077, upper_triangular, tmp41: ml1083, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1077, ml1083)

    # tmp17: ml1083, full
    # x = tmp17
    return (ml1083)
end