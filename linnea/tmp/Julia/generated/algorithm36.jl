using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm36(ml1200::Array{Float64,2}, ml1201::Array{Float64,2}, ml1202::Array{Float64,2}, ml1203::Array{Float64,2}, ml1204::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1200, full, L: ml1201, full, A: ml1202, full, B: ml1203, full, y: ml1204, full
    ml1205 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1205, ml1203)

    # R: ml1200, full, L: ml1201, full, A: ml1202, full, y: ml1204, full, tmp26: ml1205, full
    ml1206 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1202, ml1206, info) = LinearAlgebra.LAPACK.getrf!(ml1202)

    # R: ml1200, full, L: ml1201, full, y: ml1204, full, tmp26: ml1205, full, P11: ml1206, ipiv, L9: ml1202, lower_triangular_udiag, U10: ml1202, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1202, ml1205)

    # R: ml1200, full, L: ml1201, full, y: ml1204, full, P11: ml1206, ipiv, L9: ml1202, lower_triangular_udiag, tmp27: ml1205, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1202, ml1205)

    # R: ml1200, full, L: ml1201, full, y: ml1204, full, P11: ml1206, ipiv, tmp28: ml1205, full
    ml1207 = [1:length(ml1206);]
    @inbounds for i in 1:length(ml1206)
        ml1207[i], ml1207[ml1206[i]] = ml1207[ml1206[i]], ml1207[i];
    end;
    ml1208 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1208 = ml1205[invperm(ml1207),:]

    # R: ml1200, full, L: ml1201, full, y: ml1204, full, tmp25: ml1208, full
    ml1209 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1208, 0.0, ml1209)

    # R: ml1200, full, L: ml1201, full, y: ml1204, full, tmp19: ml1209, symmetric_lower_triangular
    ml1210 = diag(ml1201)
    ml1211 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1200, 1, ml1211, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1200, 2);
        view(ml1200, :, i)[:] .*= ml1210;
    end;        

    # R: ml1211, full, y: ml1204, full, tmp19: ml1209, symmetric_lower_triangular, tmp29: ml1200, full
    for i = 1:2000-1;
        view(ml1209, i, i+1:2000)[:] = view(ml1209, i+1:2000, i);
    end;
    ml1212 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1209, 1, ml1212, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1200, ml1211, 1.0, ml1209)

    # y: ml1204, full, tmp19: ml1212, full, tmp31: ml1209, full
    ml1213 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1209, ml1213, info) = LinearAlgebra.LAPACK.getrf!(ml1209)

    # y: ml1204, full, tmp19: ml1212, full, P35: ml1213, ipiv, L33: ml1209, lower_triangular_udiag, U34: ml1209, upper_triangular
    ml1214 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1212, ml1204, 0.0, ml1214)

    # P35: ml1213, ipiv, L33: ml1209, lower_triangular_udiag, U34: ml1209, upper_triangular, tmp32: ml1214, full
    ml1215 = [1:length(ml1213);]
    @inbounds for i in 1:length(ml1213)
        ml1215[i], ml1215[ml1213[i]] = ml1215[ml1213[i]], ml1215[i];
    end;
    ml1216 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1216 = ml1214[ml1215]

    # L33: ml1209, lower_triangular_udiag, U34: ml1209, upper_triangular, tmp40: ml1216, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1209, ml1216)

    # U34: ml1209, upper_triangular, tmp41: ml1216, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1209, ml1216)

    # tmp17: ml1216, full
    # x = tmp17
    return (ml1216)
end