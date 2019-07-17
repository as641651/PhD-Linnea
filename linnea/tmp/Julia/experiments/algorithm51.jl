using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm51(ml1724::Array{Float64,2}, ml1725::Array{Float64,2}, ml1726::Array{Float64,2}, ml1727::Array{Float64,2}, ml1728::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1724, full, L: ml1725, full, A: ml1726, full, B: ml1727, full, y: ml1728, full
    ml1729 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1726, ml1729, info) = LinearAlgebra.LAPACK.getrf!(ml1726)

    # R: ml1724, full, L: ml1725, full, B: ml1727, full, y: ml1728, full, P11: ml1729, ipiv, L9: ml1726, lower_triangular_udiag, U10: ml1726, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1726, ml1727)

    # R: ml1724, full, L: ml1725, full, y: ml1728, full, P11: ml1729, ipiv, L9: ml1726, lower_triangular_udiag, tmp53: ml1727, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1726, ml1727)

    # R: ml1724, full, L: ml1725, full, y: ml1728, full, P11: ml1729, ipiv, tmp54: ml1727, full
    ml1730 = [1:length(ml1729);]
    @inbounds for i in 1:length(ml1729)
        ml1730[i], ml1730[ml1729[i]] = ml1730[ml1729[i]], ml1730[i];
    end;
    ml1731 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1731 = ml1727[:,invperm(ml1730)]

    # R: ml1724, full, L: ml1725, full, y: ml1728, full, tmp55: ml1731, full
    ml1732 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1732, ml1731)

    # R: ml1724, full, L: ml1725, full, y: ml1728, full, tmp25: ml1732, full
    ml1733 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1732, 0.0, ml1733)

    # R: ml1724, full, L: ml1725, full, y: ml1728, full, tmp19: ml1733, symmetric_lower_triangular
    ml1734 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1733, ml1728, 0.0, ml1734)

    # R: ml1724, full, L: ml1725, full, tmp19: ml1733, symmetric_lower_triangular, tmp32: ml1734, full
    ml1735 = diag(ml1725)
    ml1736 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1724, 1, ml1736, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1724, 2);
        view(ml1724, :, i)[:] .*= ml1735;
    end;        

    # R: ml1736, full, tmp19: ml1733, symmetric_lower_triangular, tmp32: ml1734, full, tmp29: ml1724, full
    for i = 1:2000-1;
        view(ml1733, i, i+1:2000)[:] = view(ml1733, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1724, ml1736, 1.0, ml1733)

    # tmp32: ml1734, full, tmp31: ml1733, full
    ml1737 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1733, ml1737, info) = LinearAlgebra.LAPACK.getrf!(ml1733)

    # tmp32: ml1734, full, P35: ml1737, ipiv, L33: ml1733, lower_triangular_udiag, U34: ml1733, upper_triangular
    ml1738 = [1:length(ml1737);]
    @inbounds for i in 1:length(ml1737)
        ml1738[i], ml1738[ml1737[i]] = ml1738[ml1737[i]], ml1738[i];
    end;
    ml1739 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1739 = ml1734[ml1738]

    # L33: ml1733, lower_triangular_udiag, U34: ml1733, upper_triangular, tmp40: ml1739, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1733, ml1739)

    # U34: ml1733, upper_triangular, tmp41: ml1739, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1733, ml1739)

    # tmp17: ml1739, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1739), (finish-start)*1e-9)
end