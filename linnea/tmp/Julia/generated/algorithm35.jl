using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm35(ml1166::Array{Float64,2}, ml1167::Array{Float64,2}, ml1168::Array{Float64,2}, ml1169::Array{Float64,2}, ml1170::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1166, full, L: ml1167, full, A: ml1168, full, B: ml1169, full, y: ml1170, full
    ml1171 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1171, ml1169)

    # R: ml1166, full, L: ml1167, full, A: ml1168, full, y: ml1170, full, tmp26: ml1171, full
    ml1172 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1168, ml1172, info) = LinearAlgebra.LAPACK.getrf!(ml1168)

    # R: ml1166, full, L: ml1167, full, y: ml1170, full, tmp26: ml1171, full, P11: ml1172, ipiv, L9: ml1168, lower_triangular_udiag, U10: ml1168, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1168, ml1171)

    # R: ml1166, full, L: ml1167, full, y: ml1170, full, P11: ml1172, ipiv, L9: ml1168, lower_triangular_udiag, tmp27: ml1171, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1168, ml1171)

    # R: ml1166, full, L: ml1167, full, y: ml1170, full, P11: ml1172, ipiv, tmp28: ml1171, full
    ml1173 = [1:length(ml1172);]
    @inbounds for i in 1:length(ml1172)
        ml1173[i], ml1173[ml1172[i]] = ml1173[ml1172[i]], ml1173[i];
    end;
    ml1174 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1174 = ml1171[invperm(ml1173),:]

    # R: ml1166, full, L: ml1167, full, y: ml1170, full, tmp25: ml1174, full
    ml1175 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1174, 0.0, ml1175)

    # R: ml1166, full, L: ml1167, full, y: ml1170, full, tmp19: ml1175, symmetric_lower_triangular
    ml1176 = diag(ml1167)
    ml1177 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1166, 1, ml1177, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1166, 2);
        view(ml1166, :, i)[:] .*= ml1176;
    end;        

    # R: ml1177, full, y: ml1170, full, tmp19: ml1175, symmetric_lower_triangular, tmp29: ml1166, full
    for i = 1:2000-1;
        view(ml1175, i, i+1:2000)[:] = view(ml1175, i+1:2000, i);
    end;
    ml1178 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1175, 1, ml1178, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1166, ml1177, 1.0, ml1175)

    # y: ml1170, full, tmp19: ml1178, full, tmp31: ml1175, full
    ml1179 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1175, ml1179, info) = LinearAlgebra.LAPACK.getrf!(ml1175)

    # y: ml1170, full, tmp19: ml1178, full, P35: ml1179, ipiv, L33: ml1175, lower_triangular_udiag, U34: ml1175, upper_triangular
    ml1180 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1178, ml1170, 0.0, ml1180)

    # P35: ml1179, ipiv, L33: ml1175, lower_triangular_udiag, U34: ml1175, upper_triangular, tmp32: ml1180, full
    ml1181 = [1:length(ml1179);]
    @inbounds for i in 1:length(ml1179)
        ml1181[i], ml1181[ml1179[i]] = ml1181[ml1179[i]], ml1181[i];
    end;
    ml1182 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1182 = ml1180[ml1181]

    # L33: ml1175, lower_triangular_udiag, U34: ml1175, upper_triangular, tmp40: ml1182, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1175, ml1182)

    # U34: ml1175, upper_triangular, tmp41: ml1182, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1175, ml1182)

    # tmp17: ml1182, full
    # x = tmp17
    return (ml1182)
end