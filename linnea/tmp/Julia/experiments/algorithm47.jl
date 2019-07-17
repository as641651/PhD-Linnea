using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm47(ml1591::Array{Float64,2}, ml1592::Array{Float64,2}, ml1593::Array{Float64,2}, ml1594::Array{Float64,2}, ml1595::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1591, full, L: ml1592, full, A: ml1593, full, B: ml1594, full, y: ml1595, full
    ml1596 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1596, ml1594)

    # R: ml1591, full, L: ml1592, full, A: ml1593, full, y: ml1595, full, tmp26: ml1596, full
    ml1597 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1593, ml1597, info) = LinearAlgebra.LAPACK.getrf!(ml1593)

    # R: ml1591, full, L: ml1592, full, y: ml1595, full, tmp26: ml1596, full, P11: ml1597, ipiv, L9: ml1593, lower_triangular_udiag, U10: ml1593, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1593, ml1596)

    # R: ml1591, full, L: ml1592, full, y: ml1595, full, P11: ml1597, ipiv, L9: ml1593, lower_triangular_udiag, tmp27: ml1596, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1593, ml1596)

    # R: ml1591, full, L: ml1592, full, y: ml1595, full, P11: ml1597, ipiv, tmp28: ml1596, full
    ml1598 = [1:length(ml1597);]
    @inbounds for i in 1:length(ml1597)
        ml1598[i], ml1598[ml1597[i]] = ml1598[ml1597[i]], ml1598[i];
    end;
    ml1599 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1599 = ml1596[invperm(ml1598),:]

    # R: ml1591, full, L: ml1592, full, y: ml1595, full, tmp25: ml1599, full
    ml1600 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1599, 0.0, ml1600)

    # R: ml1591, full, L: ml1592, full, y: ml1595, full, tmp19: ml1600, symmetric_lower_triangular
    ml1601 = diag(ml1592)
    ml1602 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1591, 1, ml1602, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1591, 2);
        view(ml1591, :, i)[:] .*= ml1601;
    end;        

    # R: ml1602, full, y: ml1595, full, tmp19: ml1600, symmetric_lower_triangular, tmp29: ml1591, full
    for i = 1:2000-1;
        view(ml1600, i, i+1:2000)[:] = view(ml1600, i+1:2000, i);
    end;
    ml1603 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1600, 1, ml1603, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1591, ml1602, 1.0, ml1600)

    # y: ml1595, full, tmp19: ml1603, full, tmp31: ml1600, full
    ml1604 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1600, ml1604, info) = LinearAlgebra.LAPACK.getrf!(ml1600)

    # y: ml1595, full, tmp19: ml1603, full, P35: ml1604, ipiv, L33: ml1600, lower_triangular_udiag, U34: ml1600, upper_triangular
    ml1605 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1603, ml1595, 0.0, ml1605)

    # P35: ml1604, ipiv, L33: ml1600, lower_triangular_udiag, U34: ml1600, upper_triangular, tmp32: ml1605, full
    ml1606 = [1:length(ml1604);]
    @inbounds for i in 1:length(ml1604)
        ml1606[i], ml1606[ml1604[i]] = ml1606[ml1604[i]], ml1606[i];
    end;
    ml1607 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1607 = ml1605[ml1606]

    # L33: ml1600, lower_triangular_udiag, U34: ml1600, upper_triangular, tmp40: ml1607, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1600, ml1607)

    # U34: ml1600, upper_triangular, tmp41: ml1607, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1600, ml1607)

    # tmp17: ml1607, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1607), (finish-start)*1e-9)
end