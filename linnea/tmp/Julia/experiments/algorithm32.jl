using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm32(ml1084::Array{Float64,2}, ml1085::Array{Float64,2}, ml1086::Array{Float64,2}, ml1087::Array{Float64,2}, ml1088::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1084, full, L: ml1085, full, A: ml1086, full, B: ml1087, full, y: ml1088, full
    ml1089 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1089, ml1087)

    # R: ml1084, full, L: ml1085, full, A: ml1086, full, y: ml1088, full, tmp26: ml1089, full
    ml1090 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1086, ml1090, info) = LinearAlgebra.LAPACK.getrf!(ml1086)

    # R: ml1084, full, L: ml1085, full, y: ml1088, full, tmp26: ml1089, full, P11: ml1090, ipiv, L9: ml1086, lower_triangular_udiag, U10: ml1086, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1086, ml1089)

    # R: ml1084, full, L: ml1085, full, y: ml1088, full, P11: ml1090, ipiv, L9: ml1086, lower_triangular_udiag, tmp27: ml1089, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1086, ml1089)

    # R: ml1084, full, L: ml1085, full, y: ml1088, full, P11: ml1090, ipiv, tmp28: ml1089, full
    ml1091 = [1:length(ml1090);]
    @inbounds for i in 1:length(ml1090)
        ml1091[i], ml1091[ml1090[i]] = ml1091[ml1090[i]], ml1091[i];
    end;
    ml1092 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1092 = ml1089[invperm(ml1091),:]

    # R: ml1084, full, L: ml1085, full, y: ml1088, full, tmp25: ml1092, full
    ml1093 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1092, 0.0, ml1093)

    # R: ml1084, full, L: ml1085, full, y: ml1088, full, tmp19: ml1093, symmetric_lower_triangular
    ml1094 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1093, ml1088, 0.0, ml1094)

    # R: ml1084, full, L: ml1085, full, tmp19: ml1093, symmetric_lower_triangular, tmp32: ml1094, full
    ml1095 = diag(ml1085)
    ml1096 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1084, 1, ml1096, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1084, 2);
        view(ml1084, :, i)[:] .*= ml1095;
    end;        

    # R: ml1096, full, tmp19: ml1093, symmetric_lower_triangular, tmp32: ml1094, full, tmp29: ml1084, full
    for i = 1:2000-1;
        view(ml1093, i, i+1:2000)[:] = view(ml1093, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1096, ml1084, 1.0, ml1093)

    # tmp32: ml1094, full, tmp31: ml1093, full
    ml1097 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1093, ml1097, info) = LinearAlgebra.LAPACK.getrf!(ml1093)

    # tmp32: ml1094, full, P35: ml1097, ipiv, L33: ml1093, lower_triangular_udiag, U34: ml1093, upper_triangular
    ml1098 = [1:length(ml1097);]
    @inbounds for i in 1:length(ml1097)
        ml1098[i], ml1098[ml1097[i]] = ml1098[ml1097[i]], ml1098[i];
    end;
    ml1099 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1099 = ml1094[ml1098]

    # L33: ml1093, lower_triangular_udiag, U34: ml1093, upper_triangular, tmp40: ml1099, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1093, ml1099)

    # U34: ml1093, upper_triangular, tmp41: ml1099, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1093, ml1099)

    # tmp17: ml1099, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1099), (finish-start)*1e-9)
end