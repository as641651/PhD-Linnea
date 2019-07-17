using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm44(ml1489::Array{Float64,2}, ml1490::Array{Float64,2}, ml1491::Array{Float64,2}, ml1492::Array{Float64,2}, ml1493::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1489, full, L: ml1490, full, A: ml1491, full, B: ml1492, full, y: ml1493, full
    ml1494 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1494, ml1492)

    # R: ml1489, full, L: ml1490, full, A: ml1491, full, y: ml1493, full, tmp26: ml1494, full
    ml1495 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1491, ml1495, info) = LinearAlgebra.LAPACK.getrf!(ml1491)

    # R: ml1489, full, L: ml1490, full, y: ml1493, full, tmp26: ml1494, full, P11: ml1495, ipiv, L9: ml1491, lower_triangular_udiag, U10: ml1491, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1491, ml1494)

    # R: ml1489, full, L: ml1490, full, y: ml1493, full, P11: ml1495, ipiv, L9: ml1491, lower_triangular_udiag, tmp27: ml1494, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1491, ml1494)

    # R: ml1489, full, L: ml1490, full, y: ml1493, full, P11: ml1495, ipiv, tmp28: ml1494, full
    ml1496 = [1:length(ml1495);]
    @inbounds for i in 1:length(ml1495)
        ml1496[i], ml1496[ml1495[i]] = ml1496[ml1495[i]], ml1496[i];
    end;
    ml1497 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1497 = ml1494[invperm(ml1496),:]

    # R: ml1489, full, L: ml1490, full, y: ml1493, full, tmp25: ml1497, full
    ml1498 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1497, 0.0, ml1498)

    # R: ml1489, full, L: ml1490, full, y: ml1493, full, tmp19: ml1498, symmetric_lower_triangular
    ml1499 = diag(ml1490)
    ml1500 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1489, 1, ml1500, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1489, 2);
        view(ml1489, :, i)[:] .*= ml1499;
    end;        

    # R: ml1500, full, y: ml1493, full, tmp19: ml1498, symmetric_lower_triangular, tmp29: ml1489, full
    for i = 1:2000-1;
        view(ml1498, i, i+1:2000)[:] = view(ml1498, i+1:2000, i);
    end;
    ml1501 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1498, 1, ml1501, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1489, ml1500, 1.0, ml1498)

    # y: ml1493, full, tmp19: ml1501, full, tmp31: ml1498, full
    ml1502 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1498, ml1502, info) = LinearAlgebra.LAPACK.getrf!(ml1498)

    # y: ml1493, full, tmp19: ml1501, full, P35: ml1502, ipiv, L33: ml1498, lower_triangular_udiag, U34: ml1498, upper_triangular
    ml1503 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1501, ml1493, 0.0, ml1503)

    # P35: ml1502, ipiv, L33: ml1498, lower_triangular_udiag, U34: ml1498, upper_triangular, tmp32: ml1503, full
    ml1504 = [1:length(ml1502);]
    @inbounds for i in 1:length(ml1502)
        ml1504[i], ml1504[ml1502[i]] = ml1504[ml1502[i]], ml1504[i];
    end;
    ml1505 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1505 = ml1503[ml1504]

    # L33: ml1498, lower_triangular_udiag, U34: ml1498, upper_triangular, tmp40: ml1505, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1498, ml1505)

    # U34: ml1498, upper_triangular, tmp41: ml1505, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1498, ml1505)

    # tmp17: ml1505, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1505), (finish-start)*1e-9)
end