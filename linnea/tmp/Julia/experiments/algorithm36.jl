using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm36(ml1217::Array{Float64,2}, ml1218::Array{Float64,2}, ml1219::Array{Float64,2}, ml1220::Array{Float64,2}, ml1221::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1217, full, L: ml1218, full, A: ml1219, full, B: ml1220, full, y: ml1221, full
    ml1222 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1222, ml1220)

    # R: ml1217, full, L: ml1218, full, A: ml1219, full, y: ml1221, full, tmp26: ml1222, full
    ml1223 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1219, ml1223, info) = LinearAlgebra.LAPACK.getrf!(ml1219)

    # R: ml1217, full, L: ml1218, full, y: ml1221, full, tmp26: ml1222, full, P11: ml1223, ipiv, L9: ml1219, lower_triangular_udiag, U10: ml1219, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1219, ml1222)

    # R: ml1217, full, L: ml1218, full, y: ml1221, full, P11: ml1223, ipiv, L9: ml1219, lower_triangular_udiag, tmp27: ml1222, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1219, ml1222)

    # R: ml1217, full, L: ml1218, full, y: ml1221, full, P11: ml1223, ipiv, tmp28: ml1222, full
    ml1224 = [1:length(ml1223);]
    @inbounds for i in 1:length(ml1223)
        ml1224[i], ml1224[ml1223[i]] = ml1224[ml1223[i]], ml1224[i];
    end;
    ml1225 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1225 = ml1222[invperm(ml1224),:]

    # R: ml1217, full, L: ml1218, full, y: ml1221, full, tmp25: ml1225, full
    ml1226 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1225, 0.0, ml1226)

    # R: ml1217, full, L: ml1218, full, y: ml1221, full, tmp19: ml1226, symmetric_lower_triangular
    ml1227 = diag(ml1218)
    ml1228 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1217, 1, ml1228, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1217, 2);
        view(ml1217, :, i)[:] .*= ml1227;
    end;        

    # R: ml1228, full, y: ml1221, full, tmp19: ml1226, symmetric_lower_triangular, tmp29: ml1217, full
    for i = 1:2000-1;
        view(ml1226, i, i+1:2000)[:] = view(ml1226, i+1:2000, i);
    end;
    ml1229 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1226, 1, ml1229, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1217, ml1228, 1.0, ml1226)

    # y: ml1221, full, tmp19: ml1229, full, tmp31: ml1226, full
    ml1230 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1226, ml1230, info) = LinearAlgebra.LAPACK.getrf!(ml1226)

    # y: ml1221, full, tmp19: ml1229, full, P35: ml1230, ipiv, L33: ml1226, lower_triangular_udiag, U34: ml1226, upper_triangular
    ml1231 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1229, ml1221, 0.0, ml1231)

    # P35: ml1230, ipiv, L33: ml1226, lower_triangular_udiag, U34: ml1226, upper_triangular, tmp32: ml1231, full
    ml1232 = [1:length(ml1230);]
    @inbounds for i in 1:length(ml1230)
        ml1232[i], ml1232[ml1230[i]] = ml1232[ml1230[i]], ml1232[i];
    end;
    ml1233 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1233 = ml1231[ml1232]

    # L33: ml1226, lower_triangular_udiag, U34: ml1226, upper_triangular, tmp40: ml1233, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1226, ml1233)

    # U34: ml1226, upper_triangular, tmp41: ml1233, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1226, ml1233)

    # tmp17: ml1233, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1233), (finish-start)*1e-9)
end