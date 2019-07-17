using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm33(ml1116::Array{Float64,2}, ml1117::Array{Float64,2}, ml1118::Array{Float64,2}, ml1119::Array{Float64,2}, ml1120::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1116, full, L: ml1117, full, A: ml1118, full, B: ml1119, full, y: ml1120, full
    ml1121 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1121, ml1119)

    # R: ml1116, full, L: ml1117, full, A: ml1118, full, y: ml1120, full, tmp26: ml1121, full
    ml1122 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1118, ml1122, info) = LinearAlgebra.LAPACK.getrf!(ml1118)

    # R: ml1116, full, L: ml1117, full, y: ml1120, full, tmp26: ml1121, full, P11: ml1122, ipiv, L9: ml1118, lower_triangular_udiag, U10: ml1118, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1118, ml1121)

    # R: ml1116, full, L: ml1117, full, y: ml1120, full, P11: ml1122, ipiv, L9: ml1118, lower_triangular_udiag, tmp27: ml1121, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1118, ml1121)

    # R: ml1116, full, L: ml1117, full, y: ml1120, full, P11: ml1122, ipiv, tmp28: ml1121, full
    ml1123 = [1:length(ml1122);]
    @inbounds for i in 1:length(ml1122)
        ml1123[i], ml1123[ml1122[i]] = ml1123[ml1122[i]], ml1123[i];
    end;
    ml1124 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1124 = ml1121[invperm(ml1123),:]

    # R: ml1116, full, L: ml1117, full, y: ml1120, full, tmp25: ml1124, full
    ml1125 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1124, 0.0, ml1125)

    # R: ml1116, full, L: ml1117, full, y: ml1120, full, tmp19: ml1125, symmetric_lower_triangular
    ml1126 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1125, ml1120, 0.0, ml1126)

    # R: ml1116, full, L: ml1117, full, tmp19: ml1125, symmetric_lower_triangular, tmp32: ml1126, full
    ml1127 = diag(ml1117)
    ml1128 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1116, 1, ml1128, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1116, 2);
        view(ml1116, :, i)[:] .*= ml1127;
    end;        

    # R: ml1128, full, tmp19: ml1125, symmetric_lower_triangular, tmp32: ml1126, full, tmp29: ml1116, full
    for i = 1:2000-1;
        view(ml1125, i, i+1:2000)[:] = view(ml1125, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1128, ml1116, 1.0, ml1125)

    # tmp32: ml1126, full, tmp31: ml1125, full
    ml1129 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1125, ml1129, info) = LinearAlgebra.LAPACK.getrf!(ml1125)

    # tmp32: ml1126, full, P35: ml1129, ipiv, L33: ml1125, lower_triangular_udiag, U34: ml1125, upper_triangular
    ml1130 = [1:length(ml1129);]
    @inbounds for i in 1:length(ml1129)
        ml1130[i], ml1130[ml1129[i]] = ml1130[ml1129[i]], ml1130[i];
    end;
    ml1131 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1131 = ml1126[ml1130]

    # L33: ml1125, lower_triangular_udiag, U34: ml1125, upper_triangular, tmp40: ml1131, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1125, ml1131)

    # U34: ml1125, upper_triangular, tmp41: ml1131, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1125, ml1131)

    # tmp17: ml1131, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1131), (finish-start)*1e-9)
end