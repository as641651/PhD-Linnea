using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm49(ml1659::Array{Float64,2}, ml1660::Array{Float64,2}, ml1661::Array{Float64,2}, ml1662::Array{Float64,2}, ml1663::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1659, full, L: ml1660, full, A: ml1661, full, B: ml1662, full, y: ml1663, full
    ml1664 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1664, ml1662)

    # R: ml1659, full, L: ml1660, full, A: ml1661, full, y: ml1663, full, tmp26: ml1664, full
    ml1665 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1661, ml1665, info) = LinearAlgebra.LAPACK.getrf!(ml1661)

    # R: ml1659, full, L: ml1660, full, y: ml1663, full, tmp26: ml1664, full, P11: ml1665, ipiv, L9: ml1661, lower_triangular_udiag, U10: ml1661, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1661, ml1664)

    # R: ml1659, full, L: ml1660, full, y: ml1663, full, P11: ml1665, ipiv, L9: ml1661, lower_triangular_udiag, tmp27: ml1664, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1661, ml1664)

    # R: ml1659, full, L: ml1660, full, y: ml1663, full, P11: ml1665, ipiv, tmp28: ml1664, full
    ml1666 = [1:length(ml1665);]
    @inbounds for i in 1:length(ml1665)
        ml1666[i], ml1666[ml1665[i]] = ml1666[ml1665[i]], ml1666[i];
    end;
    ml1667 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1667 = ml1664[invperm(ml1666),:]

    # R: ml1659, full, L: ml1660, full, y: ml1663, full, tmp25: ml1667, full
    ml1668 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1667, 0.0, ml1668)

    # R: ml1659, full, L: ml1660, full, y: ml1663, full, tmp19: ml1668, symmetric_lower_triangular
    ml1669 = diag(ml1660)
    ml1670 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1659, 1, ml1670, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1659, 2);
        view(ml1659, :, i)[:] .*= ml1669;
    end;        

    # R: ml1670, full, y: ml1663, full, tmp19: ml1668, symmetric_lower_triangular, tmp29: ml1659, full
    for i = 1:2000-1;
        view(ml1668, i, i+1:2000)[:] = view(ml1668, i+1:2000, i);
    end;
    ml1671 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1668, 1, ml1671, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1659, ml1670, 1.0, ml1668)

    # y: ml1663, full, tmp19: ml1671, full, tmp31: ml1668, full
    ml1672 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1668, ml1672, info) = LinearAlgebra.LAPACK.getrf!(ml1668)

    # y: ml1663, full, tmp19: ml1671, full, P35: ml1672, ipiv, L33: ml1668, lower_triangular_udiag, U34: ml1668, upper_triangular
    ml1673 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1671, ml1663, 0.0, ml1673)

    # P35: ml1672, ipiv, L33: ml1668, lower_triangular_udiag, U34: ml1668, upper_triangular, tmp32: ml1673, full
    ml1674 = [1:length(ml1672);]
    @inbounds for i in 1:length(ml1672)
        ml1674[i], ml1674[ml1672[i]] = ml1674[ml1672[i]], ml1674[i];
    end;
    ml1675 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1675 = ml1673[ml1674]

    # L33: ml1668, lower_triangular_udiag, U34: ml1668, upper_triangular, tmp40: ml1675, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1668, ml1675)

    # U34: ml1668, upper_triangular, tmp41: ml1675, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1668, ml1675)

    # tmp17: ml1675, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1675), (finish-start)*1e-9)
end