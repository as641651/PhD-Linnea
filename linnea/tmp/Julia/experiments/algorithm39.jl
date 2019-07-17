using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm39(ml1319::Array{Float64,2}, ml1320::Array{Float64,2}, ml1321::Array{Float64,2}, ml1322::Array{Float64,2}, ml1323::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1319, full, L: ml1320, full, A: ml1321, full, B: ml1322, full, y: ml1323, full
    ml1324 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1324, ml1322)

    # R: ml1319, full, L: ml1320, full, A: ml1321, full, y: ml1323, full, tmp26: ml1324, full
    ml1325 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1321, ml1325, info) = LinearAlgebra.LAPACK.getrf!(ml1321)

    # R: ml1319, full, L: ml1320, full, y: ml1323, full, tmp26: ml1324, full, P11: ml1325, ipiv, L9: ml1321, lower_triangular_udiag, U10: ml1321, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1321, ml1324)

    # R: ml1319, full, L: ml1320, full, y: ml1323, full, P11: ml1325, ipiv, L9: ml1321, lower_triangular_udiag, tmp27: ml1324, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1321, ml1324)

    # R: ml1319, full, L: ml1320, full, y: ml1323, full, P11: ml1325, ipiv, tmp28: ml1324, full
    ml1326 = [1:length(ml1325);]
    @inbounds for i in 1:length(ml1325)
        ml1326[i], ml1326[ml1325[i]] = ml1326[ml1325[i]], ml1326[i];
    end;
    ml1327 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1327 = ml1324[invperm(ml1326),:]

    # R: ml1319, full, L: ml1320, full, y: ml1323, full, tmp25: ml1327, full
    ml1328 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1327, 0.0, ml1328)

    # R: ml1319, full, L: ml1320, full, y: ml1323, full, tmp19: ml1328, symmetric_lower_triangular
    ml1329 = diag(ml1320)
    ml1330 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1319, 1, ml1330, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1319, 2);
        view(ml1319, :, i)[:] .*= ml1329;
    end;        

    # R: ml1330, full, y: ml1323, full, tmp19: ml1328, symmetric_lower_triangular, tmp29: ml1319, full
    for i = 1:2000-1;
        view(ml1328, i, i+1:2000)[:] = view(ml1328, i+1:2000, i);
    end;
    ml1331 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1328, 1, ml1331, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1319, ml1330, 1.0, ml1328)

    # y: ml1323, full, tmp19: ml1331, full, tmp31: ml1328, full
    ml1332 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1328, ml1332, info) = LinearAlgebra.LAPACK.getrf!(ml1328)

    # y: ml1323, full, tmp19: ml1331, full, P35: ml1332, ipiv, L33: ml1328, lower_triangular_udiag, U34: ml1328, upper_triangular
    ml1333 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1331, ml1323, 0.0, ml1333)

    # P35: ml1332, ipiv, L33: ml1328, lower_triangular_udiag, U34: ml1328, upper_triangular, tmp32: ml1333, full
    ml1334 = [1:length(ml1332);]
    @inbounds for i in 1:length(ml1332)
        ml1334[i], ml1334[ml1332[i]] = ml1334[ml1332[i]], ml1334[i];
    end;
    ml1335 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1335 = ml1333[ml1334]

    # L33: ml1328, lower_triangular_udiag, U34: ml1328, upper_triangular, tmp40: ml1335, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1328, ml1335)

    # U34: ml1328, upper_triangular, tmp41: ml1335, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1328, ml1335)

    # tmp17: ml1335, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1335), (finish-start)*1e-9)
end