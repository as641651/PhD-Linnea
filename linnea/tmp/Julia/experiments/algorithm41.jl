using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm41(ml1387::Array{Float64,2}, ml1388::Array{Float64,2}, ml1389::Array{Float64,2}, ml1390::Array{Float64,2}, ml1391::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1387, full, L: ml1388, full, A: ml1389, full, B: ml1390, full, y: ml1391, full
    ml1392 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1392, ml1390)

    # R: ml1387, full, L: ml1388, full, A: ml1389, full, y: ml1391, full, tmp26: ml1392, full
    ml1393 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1389, ml1393, info) = LinearAlgebra.LAPACK.getrf!(ml1389)

    # R: ml1387, full, L: ml1388, full, y: ml1391, full, tmp26: ml1392, full, P11: ml1393, ipiv, L9: ml1389, lower_triangular_udiag, U10: ml1389, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1389, ml1392)

    # R: ml1387, full, L: ml1388, full, y: ml1391, full, P11: ml1393, ipiv, L9: ml1389, lower_triangular_udiag, tmp27: ml1392, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1389, ml1392)

    # R: ml1387, full, L: ml1388, full, y: ml1391, full, P11: ml1393, ipiv, tmp28: ml1392, full
    ml1394 = [1:length(ml1393);]
    @inbounds for i in 1:length(ml1393)
        ml1394[i], ml1394[ml1393[i]] = ml1394[ml1393[i]], ml1394[i];
    end;
    ml1395 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1395 = ml1392[invperm(ml1394),:]

    # R: ml1387, full, L: ml1388, full, y: ml1391, full, tmp25: ml1395, full
    ml1396 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1395, 0.0, ml1396)

    # R: ml1387, full, L: ml1388, full, y: ml1391, full, tmp19: ml1396, symmetric_lower_triangular
    ml1397 = diag(ml1388)
    ml1398 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1387, 1, ml1398, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1387, 2);
        view(ml1387, :, i)[:] .*= ml1397;
    end;        

    # R: ml1398, full, y: ml1391, full, tmp19: ml1396, symmetric_lower_triangular, tmp29: ml1387, full
    for i = 1:2000-1;
        view(ml1396, i, i+1:2000)[:] = view(ml1396, i+1:2000, i);
    end;
    ml1399 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1396, 1, ml1399, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1387, ml1398, 1.0, ml1396)

    # y: ml1391, full, tmp19: ml1399, full, tmp31: ml1396, full
    ml1400 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1399, ml1391, 0.0, ml1400)

    # tmp31: ml1396, full, tmp32: ml1400, full
    ml1401 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1396, ml1401, info) = LinearAlgebra.LAPACK.getrf!(ml1396)

    # tmp32: ml1400, full, P35: ml1401, ipiv, L33: ml1396, lower_triangular_udiag, U34: ml1396, upper_triangular
    ml1402 = [1:length(ml1401);]
    @inbounds for i in 1:length(ml1401)
        ml1402[i], ml1402[ml1401[i]] = ml1402[ml1401[i]], ml1402[i];
    end;
    ml1403 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1403 = ml1400[ml1402]

    # L33: ml1396, lower_triangular_udiag, U34: ml1396, upper_triangular, tmp40: ml1403, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1396, ml1403)

    # U34: ml1396, upper_triangular, tmp41: ml1403, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1396, ml1403)

    # tmp17: ml1403, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1403), (finish-start)*1e-9)
end