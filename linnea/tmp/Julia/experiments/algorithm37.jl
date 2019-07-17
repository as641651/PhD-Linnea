using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm37(ml1251::Array{Float64,2}, ml1252::Array{Float64,2}, ml1253::Array{Float64,2}, ml1254::Array{Float64,2}, ml1255::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1251, full, L: ml1252, full, A: ml1253, full, B: ml1254, full, y: ml1255, full
    ml1256 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1256, ml1254)

    # R: ml1251, full, L: ml1252, full, A: ml1253, full, y: ml1255, full, tmp26: ml1256, full
    ml1257 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1253, ml1257, info) = LinearAlgebra.LAPACK.getrf!(ml1253)

    # R: ml1251, full, L: ml1252, full, y: ml1255, full, tmp26: ml1256, full, P11: ml1257, ipiv, L9: ml1253, lower_triangular_udiag, U10: ml1253, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1253, ml1256)

    # R: ml1251, full, L: ml1252, full, y: ml1255, full, P11: ml1257, ipiv, L9: ml1253, lower_triangular_udiag, tmp27: ml1256, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1253, ml1256)

    # R: ml1251, full, L: ml1252, full, y: ml1255, full, P11: ml1257, ipiv, tmp28: ml1256, full
    ml1258 = [1:length(ml1257);]
    @inbounds for i in 1:length(ml1257)
        ml1258[i], ml1258[ml1257[i]] = ml1258[ml1257[i]], ml1258[i];
    end;
    ml1259 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1259 = ml1256[invperm(ml1258),:]

    # R: ml1251, full, L: ml1252, full, y: ml1255, full, tmp25: ml1259, full
    ml1260 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1259, 0.0, ml1260)

    # R: ml1251, full, L: ml1252, full, y: ml1255, full, tmp19: ml1260, symmetric_lower_triangular
    ml1261 = diag(ml1252)
    ml1262 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1251, 1, ml1262, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1251, 2);
        view(ml1251, :, i)[:] .*= ml1261;
    end;        

    # R: ml1262, full, y: ml1255, full, tmp19: ml1260, symmetric_lower_triangular, tmp29: ml1251, full
    for i = 1:2000-1;
        view(ml1260, i, i+1:2000)[:] = view(ml1260, i+1:2000, i);
    end;
    ml1263 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1260, 1, ml1263, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1251, ml1262, 1.0, ml1260)

    # y: ml1255, full, tmp19: ml1263, full, tmp31: ml1260, full
    ml1264 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1260, ml1264, info) = LinearAlgebra.LAPACK.getrf!(ml1260)

    # y: ml1255, full, tmp19: ml1263, full, P35: ml1264, ipiv, L33: ml1260, lower_triangular_udiag, U34: ml1260, upper_triangular
    ml1265 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1263, ml1255, 0.0, ml1265)

    # P35: ml1264, ipiv, L33: ml1260, lower_triangular_udiag, U34: ml1260, upper_triangular, tmp32: ml1265, full
    ml1266 = [1:length(ml1264);]
    @inbounds for i in 1:length(ml1264)
        ml1266[i], ml1266[ml1264[i]] = ml1266[ml1264[i]], ml1266[i];
    end;
    ml1267 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1267 = ml1265[ml1266]

    # L33: ml1260, lower_triangular_udiag, U34: ml1260, upper_triangular, tmp40: ml1267, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1260, ml1267)

    # U34: ml1260, upper_triangular, tmp41: ml1267, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1260, ml1267)

    # tmp17: ml1267, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1267), (finish-start)*1e-9)
end