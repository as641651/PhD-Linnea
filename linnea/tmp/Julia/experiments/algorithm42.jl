using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm42(ml1421::Array{Float64,2}, ml1422::Array{Float64,2}, ml1423::Array{Float64,2}, ml1424::Array{Float64,2}, ml1425::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1421, full, L: ml1422, full, A: ml1423, full, B: ml1424, full, y: ml1425, full
    ml1426 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1426, ml1424)

    # R: ml1421, full, L: ml1422, full, A: ml1423, full, y: ml1425, full, tmp26: ml1426, full
    ml1427 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1423, ml1427, info) = LinearAlgebra.LAPACK.getrf!(ml1423)

    # R: ml1421, full, L: ml1422, full, y: ml1425, full, tmp26: ml1426, full, P11: ml1427, ipiv, L9: ml1423, lower_triangular_udiag, U10: ml1423, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1423, ml1426)

    # R: ml1421, full, L: ml1422, full, y: ml1425, full, P11: ml1427, ipiv, L9: ml1423, lower_triangular_udiag, tmp27: ml1426, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1423, ml1426)

    # R: ml1421, full, L: ml1422, full, y: ml1425, full, P11: ml1427, ipiv, tmp28: ml1426, full
    ml1428 = [1:length(ml1427);]
    @inbounds for i in 1:length(ml1427)
        ml1428[i], ml1428[ml1427[i]] = ml1428[ml1427[i]], ml1428[i];
    end;
    ml1429 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1429 = ml1426[invperm(ml1428),:]

    # R: ml1421, full, L: ml1422, full, y: ml1425, full, tmp25: ml1429, full
    ml1430 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1429, 0.0, ml1430)

    # R: ml1421, full, L: ml1422, full, y: ml1425, full, tmp19: ml1430, symmetric_lower_triangular
    ml1431 = diag(ml1422)
    ml1432 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1421, 1, ml1432, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1421, 2);
        view(ml1421, :, i)[:] .*= ml1431;
    end;        

    # R: ml1432, full, y: ml1425, full, tmp19: ml1430, symmetric_lower_triangular, tmp29: ml1421, full
    for i = 1:2000-1;
        view(ml1430, i, i+1:2000)[:] = view(ml1430, i+1:2000, i);
    end;
    ml1433 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1430, 1, ml1433, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1421, ml1432, 1.0, ml1430)

    # y: ml1425, full, tmp19: ml1433, full, tmp31: ml1430, full
    ml1434 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1433, ml1425, 0.0, ml1434)

    # tmp31: ml1430, full, tmp32: ml1434, full
    ml1435 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1430, ml1435, info) = LinearAlgebra.LAPACK.getrf!(ml1430)

    # tmp32: ml1434, full, P35: ml1435, ipiv, L33: ml1430, lower_triangular_udiag, U34: ml1430, upper_triangular
    ml1436 = [1:length(ml1435);]
    @inbounds for i in 1:length(ml1435)
        ml1436[i], ml1436[ml1435[i]] = ml1436[ml1435[i]], ml1436[i];
    end;
    ml1437 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1437 = ml1434[ml1436]

    # L33: ml1430, lower_triangular_udiag, U34: ml1430, upper_triangular, tmp40: ml1437, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1430, ml1437)

    # U34: ml1430, upper_triangular, tmp41: ml1437, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1430, ml1437)

    # tmp17: ml1437, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1437), (finish-start)*1e-9)
end