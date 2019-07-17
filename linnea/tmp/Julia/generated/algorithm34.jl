using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm34(ml1132::Array{Float64,2}, ml1133::Array{Float64,2}, ml1134::Array{Float64,2}, ml1135::Array{Float64,2}, ml1136::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1132, full, L: ml1133, full, A: ml1134, full, B: ml1135, full, y: ml1136, full
    ml1137 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1137, ml1135)

    # R: ml1132, full, L: ml1133, full, A: ml1134, full, y: ml1136, full, tmp26: ml1137, full
    ml1138 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1134, ml1138, info) = LinearAlgebra.LAPACK.getrf!(ml1134)

    # R: ml1132, full, L: ml1133, full, y: ml1136, full, tmp26: ml1137, full, P11: ml1138, ipiv, L9: ml1134, lower_triangular_udiag, U10: ml1134, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1134, ml1137)

    # R: ml1132, full, L: ml1133, full, y: ml1136, full, P11: ml1138, ipiv, L9: ml1134, lower_triangular_udiag, tmp27: ml1137, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1134, ml1137)

    # R: ml1132, full, L: ml1133, full, y: ml1136, full, P11: ml1138, ipiv, tmp28: ml1137, full
    ml1139 = [1:length(ml1138);]
    @inbounds for i in 1:length(ml1138)
        ml1139[i], ml1139[ml1138[i]] = ml1139[ml1138[i]], ml1139[i];
    end;
    ml1140 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1140 = ml1137[invperm(ml1139),:]

    # R: ml1132, full, L: ml1133, full, y: ml1136, full, tmp25: ml1140, full
    ml1141 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1140, 0.0, ml1141)

    # R: ml1132, full, L: ml1133, full, y: ml1136, full, tmp19: ml1141, symmetric_lower_triangular
    ml1142 = diag(ml1133)
    ml1143 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1132, 1, ml1143, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1132, 2);
        view(ml1132, :, i)[:] .*= ml1142;
    end;        

    # R: ml1143, full, y: ml1136, full, tmp19: ml1141, symmetric_lower_triangular, tmp29: ml1132, full
    for i = 1:2000-1;
        view(ml1141, i, i+1:2000)[:] = view(ml1141, i+1:2000, i);
    end;
    ml1144 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1141, 1, ml1144, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1132, ml1143, 1.0, ml1141)

    # y: ml1136, full, tmp19: ml1144, full, tmp31: ml1141, full
    ml1145 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1141, ml1145, info) = LinearAlgebra.LAPACK.getrf!(ml1141)

    # y: ml1136, full, tmp19: ml1144, full, P35: ml1145, ipiv, L33: ml1141, lower_triangular_udiag, U34: ml1141, upper_triangular
    ml1146 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1144, ml1136, 0.0, ml1146)

    # P35: ml1145, ipiv, L33: ml1141, lower_triangular_udiag, U34: ml1141, upper_triangular, tmp32: ml1146, full
    ml1147 = [1:length(ml1145);]
    @inbounds for i in 1:length(ml1145)
        ml1147[i], ml1147[ml1145[i]] = ml1147[ml1145[i]], ml1147[i];
    end;
    ml1148 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1148 = ml1146[ml1147]

    # L33: ml1141, lower_triangular_udiag, U34: ml1141, upper_triangular, tmp40: ml1148, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1141, ml1148)

    # U34: ml1141, upper_triangular, tmp41: ml1148, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1141, ml1148)

    # tmp17: ml1148, full
    # x = tmp17
    return (ml1148)
end