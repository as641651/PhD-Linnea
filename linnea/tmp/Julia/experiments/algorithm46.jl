using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm46(ml1557::Array{Float64,2}, ml1558::Array{Float64,2}, ml1559::Array{Float64,2}, ml1560::Array{Float64,2}, ml1561::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1557, full, L: ml1558, full, A: ml1559, full, B: ml1560, full, y: ml1561, full
    ml1562 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1562, ml1560)

    # R: ml1557, full, L: ml1558, full, A: ml1559, full, y: ml1561, full, tmp26: ml1562, full
    ml1563 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1559, ml1563, info) = LinearAlgebra.LAPACK.getrf!(ml1559)

    # R: ml1557, full, L: ml1558, full, y: ml1561, full, tmp26: ml1562, full, P11: ml1563, ipiv, L9: ml1559, lower_triangular_udiag, U10: ml1559, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1559, ml1562)

    # R: ml1557, full, L: ml1558, full, y: ml1561, full, P11: ml1563, ipiv, L9: ml1559, lower_triangular_udiag, tmp27: ml1562, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1559, ml1562)

    # R: ml1557, full, L: ml1558, full, y: ml1561, full, P11: ml1563, ipiv, tmp28: ml1562, full
    ml1564 = [1:length(ml1563);]
    @inbounds for i in 1:length(ml1563)
        ml1564[i], ml1564[ml1563[i]] = ml1564[ml1563[i]], ml1564[i];
    end;
    ml1565 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1565 = ml1562[invperm(ml1564),:]

    # R: ml1557, full, L: ml1558, full, y: ml1561, full, tmp25: ml1565, full
    ml1566 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1565, 0.0, ml1566)

    # R: ml1557, full, L: ml1558, full, y: ml1561, full, tmp19: ml1566, symmetric_lower_triangular
    ml1567 = diag(ml1558)
    ml1568 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1557, 1, ml1568, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1557, 2);
        view(ml1557, :, i)[:] .*= ml1567;
    end;        

    # R: ml1568, full, y: ml1561, full, tmp19: ml1566, symmetric_lower_triangular, tmp29: ml1557, full
    for i = 1:2000-1;
        view(ml1566, i, i+1:2000)[:] = view(ml1566, i+1:2000, i);
    end;
    ml1569 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1566, 1, ml1569, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1557, ml1568, 1.0, ml1566)

    # y: ml1561, full, tmp19: ml1569, full, tmp31: ml1566, full
    ml1570 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1566, ml1570, info) = LinearAlgebra.LAPACK.getrf!(ml1566)

    # y: ml1561, full, tmp19: ml1569, full, P35: ml1570, ipiv, L33: ml1566, lower_triangular_udiag, U34: ml1566, upper_triangular
    ml1571 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1569, ml1561, 0.0, ml1571)

    # P35: ml1570, ipiv, L33: ml1566, lower_triangular_udiag, U34: ml1566, upper_triangular, tmp32: ml1571, full
    ml1572 = [1:length(ml1570);]
    @inbounds for i in 1:length(ml1570)
        ml1572[i], ml1572[ml1570[i]] = ml1572[ml1570[i]], ml1572[i];
    end;
    ml1573 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1573 = ml1571[ml1572]

    # L33: ml1566, lower_triangular_udiag, U34: ml1566, upper_triangular, tmp40: ml1573, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1566, ml1573)

    # U34: ml1566, upper_triangular, tmp41: ml1573, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1566, ml1573)

    # tmp17: ml1573, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1573), (finish-start)*1e-9)
end