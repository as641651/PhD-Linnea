using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm34(ml1149::Array{Float64,2}, ml1150::Array{Float64,2}, ml1151::Array{Float64,2}, ml1152::Array{Float64,2}, ml1153::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1149, full, L: ml1150, full, A: ml1151, full, B: ml1152, full, y: ml1153, full
    ml1154 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1154, ml1152)

    # R: ml1149, full, L: ml1150, full, A: ml1151, full, y: ml1153, full, tmp26: ml1154, full
    ml1155 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1151, ml1155, info) = LinearAlgebra.LAPACK.getrf!(ml1151)

    # R: ml1149, full, L: ml1150, full, y: ml1153, full, tmp26: ml1154, full, P11: ml1155, ipiv, L9: ml1151, lower_triangular_udiag, U10: ml1151, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1151, ml1154)

    # R: ml1149, full, L: ml1150, full, y: ml1153, full, P11: ml1155, ipiv, L9: ml1151, lower_triangular_udiag, tmp27: ml1154, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1151, ml1154)

    # R: ml1149, full, L: ml1150, full, y: ml1153, full, P11: ml1155, ipiv, tmp28: ml1154, full
    ml1156 = [1:length(ml1155);]
    @inbounds for i in 1:length(ml1155)
        ml1156[i], ml1156[ml1155[i]] = ml1156[ml1155[i]], ml1156[i];
    end;
    ml1157 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1157 = ml1154[invperm(ml1156),:]

    # R: ml1149, full, L: ml1150, full, y: ml1153, full, tmp25: ml1157, full
    ml1158 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1157, 0.0, ml1158)

    # R: ml1149, full, L: ml1150, full, y: ml1153, full, tmp19: ml1158, symmetric_lower_triangular
    ml1159 = diag(ml1150)
    ml1160 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1149, 1, ml1160, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1149, 2);
        view(ml1149, :, i)[:] .*= ml1159;
    end;        

    # R: ml1160, full, y: ml1153, full, tmp19: ml1158, symmetric_lower_triangular, tmp29: ml1149, full
    for i = 1:2000-1;
        view(ml1158, i, i+1:2000)[:] = view(ml1158, i+1:2000, i);
    end;
    ml1161 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1158, 1, ml1161, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1149, ml1160, 1.0, ml1158)

    # y: ml1153, full, tmp19: ml1161, full, tmp31: ml1158, full
    ml1162 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1158, ml1162, info) = LinearAlgebra.LAPACK.getrf!(ml1158)

    # y: ml1153, full, tmp19: ml1161, full, P35: ml1162, ipiv, L33: ml1158, lower_triangular_udiag, U34: ml1158, upper_triangular
    ml1163 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1161, ml1153, 0.0, ml1163)

    # P35: ml1162, ipiv, L33: ml1158, lower_triangular_udiag, U34: ml1158, upper_triangular, tmp32: ml1163, full
    ml1164 = [1:length(ml1162);]
    @inbounds for i in 1:length(ml1162)
        ml1164[i], ml1164[ml1162[i]] = ml1164[ml1162[i]], ml1164[i];
    end;
    ml1165 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1165 = ml1163[ml1164]

    # L33: ml1158, lower_triangular_udiag, U34: ml1158, upper_triangular, tmp40: ml1165, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1158, ml1165)

    # U34: ml1158, upper_triangular, tmp41: ml1165, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1158, ml1165)

    # tmp17: ml1165, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1165), (finish-start)*1e-9)
end