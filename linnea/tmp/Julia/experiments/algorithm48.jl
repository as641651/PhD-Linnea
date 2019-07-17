using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm48(ml1625::Array{Float64,2}, ml1626::Array{Float64,2}, ml1627::Array{Float64,2}, ml1628::Array{Float64,2}, ml1629::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1625, full, L: ml1626, full, A: ml1627, full, B: ml1628, full, y: ml1629, full
    ml1630 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1630, ml1628)

    # R: ml1625, full, L: ml1626, full, A: ml1627, full, y: ml1629, full, tmp26: ml1630, full
    ml1631 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1627, ml1631, info) = LinearAlgebra.LAPACK.getrf!(ml1627)

    # R: ml1625, full, L: ml1626, full, y: ml1629, full, tmp26: ml1630, full, P11: ml1631, ipiv, L9: ml1627, lower_triangular_udiag, U10: ml1627, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1627, ml1630)

    # R: ml1625, full, L: ml1626, full, y: ml1629, full, P11: ml1631, ipiv, L9: ml1627, lower_triangular_udiag, tmp27: ml1630, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1627, ml1630)

    # R: ml1625, full, L: ml1626, full, y: ml1629, full, P11: ml1631, ipiv, tmp28: ml1630, full
    ml1632 = [1:length(ml1631);]
    @inbounds for i in 1:length(ml1631)
        ml1632[i], ml1632[ml1631[i]] = ml1632[ml1631[i]], ml1632[i];
    end;
    ml1633 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1633 = ml1630[invperm(ml1632),:]

    # R: ml1625, full, L: ml1626, full, y: ml1629, full, tmp25: ml1633, full
    ml1634 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1633, 0.0, ml1634)

    # R: ml1625, full, L: ml1626, full, y: ml1629, full, tmp19: ml1634, symmetric_lower_triangular
    ml1635 = diag(ml1626)
    ml1636 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1625, 1, ml1636, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1625, 2);
        view(ml1625, :, i)[:] .*= ml1635;
    end;        

    # R: ml1636, full, y: ml1629, full, tmp19: ml1634, symmetric_lower_triangular, tmp29: ml1625, full
    for i = 1:2000-1;
        view(ml1634, i, i+1:2000)[:] = view(ml1634, i+1:2000, i);
    end;
    ml1637 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1634, 1, ml1637, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1625, ml1636, 1.0, ml1634)

    # y: ml1629, full, tmp19: ml1637, full, tmp31: ml1634, full
    ml1638 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1634, ml1638, info) = LinearAlgebra.LAPACK.getrf!(ml1634)

    # y: ml1629, full, tmp19: ml1637, full, P35: ml1638, ipiv, L33: ml1634, lower_triangular_udiag, U34: ml1634, upper_triangular
    ml1639 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1637, ml1629, 0.0, ml1639)

    # P35: ml1638, ipiv, L33: ml1634, lower_triangular_udiag, U34: ml1634, upper_triangular, tmp32: ml1639, full
    ml1640 = [1:length(ml1638);]
    @inbounds for i in 1:length(ml1638)
        ml1640[i], ml1640[ml1638[i]] = ml1640[ml1638[i]], ml1640[i];
    end;
    ml1641 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1641 = ml1639[ml1640]

    # L33: ml1634, lower_triangular_udiag, U34: ml1634, upper_triangular, tmp40: ml1641, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1634, ml1641)

    # U34: ml1634, upper_triangular, tmp41: ml1641, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1634, ml1641)

    # tmp17: ml1641, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1641), (finish-start)*1e-9)
end