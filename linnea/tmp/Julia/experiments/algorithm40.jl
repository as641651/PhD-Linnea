using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm40(ml1353::Array{Float64,2}, ml1354::Array{Float64,2}, ml1355::Array{Float64,2}, ml1356::Array{Float64,2}, ml1357::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1353, full, L: ml1354, full, A: ml1355, full, B: ml1356, full, y: ml1357, full
    ml1358 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1358, ml1356)

    # R: ml1353, full, L: ml1354, full, A: ml1355, full, y: ml1357, full, tmp26: ml1358, full
    ml1359 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1355, ml1359, info) = LinearAlgebra.LAPACK.getrf!(ml1355)

    # R: ml1353, full, L: ml1354, full, y: ml1357, full, tmp26: ml1358, full, P11: ml1359, ipiv, L9: ml1355, lower_triangular_udiag, U10: ml1355, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1355, ml1358)

    # R: ml1353, full, L: ml1354, full, y: ml1357, full, P11: ml1359, ipiv, L9: ml1355, lower_triangular_udiag, tmp27: ml1358, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1355, ml1358)

    # R: ml1353, full, L: ml1354, full, y: ml1357, full, P11: ml1359, ipiv, tmp28: ml1358, full
    ml1360 = [1:length(ml1359);]
    @inbounds for i in 1:length(ml1359)
        ml1360[i], ml1360[ml1359[i]] = ml1360[ml1359[i]], ml1360[i];
    end;
    ml1361 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1361 = ml1358[invperm(ml1360),:]

    # R: ml1353, full, L: ml1354, full, y: ml1357, full, tmp25: ml1361, full
    ml1362 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1361, 0.0, ml1362)

    # R: ml1353, full, L: ml1354, full, y: ml1357, full, tmp19: ml1362, symmetric_lower_triangular
    ml1363 = diag(ml1354)
    ml1364 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1353, 1, ml1364, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1353, 2);
        view(ml1353, :, i)[:] .*= ml1363;
    end;        

    # R: ml1364, full, y: ml1357, full, tmp19: ml1362, symmetric_lower_triangular, tmp29: ml1353, full
    for i = 1:2000-1;
        view(ml1362, i, i+1:2000)[:] = view(ml1362, i+1:2000, i);
    end;
    ml1365 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1362, 1, ml1365, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1353, ml1364, 1.0, ml1362)

    # y: ml1357, full, tmp19: ml1365, full, tmp31: ml1362, full
    ml1366 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1365, ml1357, 0.0, ml1366)

    # tmp31: ml1362, full, tmp32: ml1366, full
    ml1367 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1362, ml1367, info) = LinearAlgebra.LAPACK.getrf!(ml1362)

    # tmp32: ml1366, full, P35: ml1367, ipiv, L33: ml1362, lower_triangular_udiag, U34: ml1362, upper_triangular
    ml1368 = [1:length(ml1367);]
    @inbounds for i in 1:length(ml1367)
        ml1368[i], ml1368[ml1367[i]] = ml1368[ml1367[i]], ml1368[i];
    end;
    ml1369 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1369 = ml1366[ml1368]

    # L33: ml1362, lower_triangular_udiag, U34: ml1362, upper_triangular, tmp40: ml1369, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1362, ml1369)

    # U34: ml1362, upper_triangular, tmp41: ml1369, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1362, ml1369)

    # tmp17: ml1369, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1369), (finish-start)*1e-9)
end