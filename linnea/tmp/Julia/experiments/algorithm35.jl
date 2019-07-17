using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm35(ml1183::Array{Float64,2}, ml1184::Array{Float64,2}, ml1185::Array{Float64,2}, ml1186::Array{Float64,2}, ml1187::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1183, full, L: ml1184, full, A: ml1185, full, B: ml1186, full, y: ml1187, full
    ml1188 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1188, ml1186)

    # R: ml1183, full, L: ml1184, full, A: ml1185, full, y: ml1187, full, tmp26: ml1188, full
    ml1189 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1185, ml1189, info) = LinearAlgebra.LAPACK.getrf!(ml1185)

    # R: ml1183, full, L: ml1184, full, y: ml1187, full, tmp26: ml1188, full, P11: ml1189, ipiv, L9: ml1185, lower_triangular_udiag, U10: ml1185, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1185, ml1188)

    # R: ml1183, full, L: ml1184, full, y: ml1187, full, P11: ml1189, ipiv, L9: ml1185, lower_triangular_udiag, tmp27: ml1188, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1185, ml1188)

    # R: ml1183, full, L: ml1184, full, y: ml1187, full, P11: ml1189, ipiv, tmp28: ml1188, full
    ml1190 = [1:length(ml1189);]
    @inbounds for i in 1:length(ml1189)
        ml1190[i], ml1190[ml1189[i]] = ml1190[ml1189[i]], ml1190[i];
    end;
    ml1191 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1191 = ml1188[invperm(ml1190),:]

    # R: ml1183, full, L: ml1184, full, y: ml1187, full, tmp25: ml1191, full
    ml1192 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1191, 0.0, ml1192)

    # R: ml1183, full, L: ml1184, full, y: ml1187, full, tmp19: ml1192, symmetric_lower_triangular
    ml1193 = diag(ml1184)
    ml1194 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1183, 1, ml1194, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1183, 2);
        view(ml1183, :, i)[:] .*= ml1193;
    end;        

    # R: ml1194, full, y: ml1187, full, tmp19: ml1192, symmetric_lower_triangular, tmp29: ml1183, full
    for i = 1:2000-1;
        view(ml1192, i, i+1:2000)[:] = view(ml1192, i+1:2000, i);
    end;
    ml1195 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1192, 1, ml1195, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1183, ml1194, 1.0, ml1192)

    # y: ml1187, full, tmp19: ml1195, full, tmp31: ml1192, full
    ml1196 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1192, ml1196, info) = LinearAlgebra.LAPACK.getrf!(ml1192)

    # y: ml1187, full, tmp19: ml1195, full, P35: ml1196, ipiv, L33: ml1192, lower_triangular_udiag, U34: ml1192, upper_triangular
    ml1197 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1195, ml1187, 0.0, ml1197)

    # P35: ml1196, ipiv, L33: ml1192, lower_triangular_udiag, U34: ml1192, upper_triangular, tmp32: ml1197, full
    ml1198 = [1:length(ml1196);]
    @inbounds for i in 1:length(ml1196)
        ml1198[i], ml1198[ml1196[i]] = ml1198[ml1196[i]], ml1198[i];
    end;
    ml1199 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1199 = ml1197[ml1198]

    # L33: ml1192, lower_triangular_udiag, U34: ml1192, upper_triangular, tmp40: ml1199, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1192, ml1199)

    # U34: ml1192, upper_triangular, tmp41: ml1199, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1192, ml1199)

    # tmp17: ml1199, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1199), (finish-start)*1e-9)
end