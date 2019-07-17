using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm53(ml1788::Array{Float64,2}, ml1789::Array{Float64,2}, ml1790::Array{Float64,2}, ml1791::Array{Float64,2}, ml1792::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1788, full, L: ml1789, full, A: ml1790, full, B: ml1791, full, y: ml1792, full
    ml1793 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1790, ml1793, info) = LinearAlgebra.LAPACK.getrf!(ml1790)

    # R: ml1788, full, L: ml1789, full, B: ml1791, full, y: ml1792, full, P11: ml1793, ipiv, L9: ml1790, lower_triangular_udiag, U10: ml1790, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1790, ml1791)

    # R: ml1788, full, L: ml1789, full, y: ml1792, full, P11: ml1793, ipiv, L9: ml1790, lower_triangular_udiag, tmp53: ml1791, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1790, ml1791)

    # R: ml1788, full, L: ml1789, full, y: ml1792, full, P11: ml1793, ipiv, tmp54: ml1791, full
    ml1794 = [1:length(ml1793);]
    @inbounds for i in 1:length(ml1793)
        ml1794[i], ml1794[ml1793[i]] = ml1794[ml1793[i]], ml1794[i];
    end;
    ml1795 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1795 = ml1791[:,invperm(ml1794)]

    # R: ml1788, full, L: ml1789, full, y: ml1792, full, tmp55: ml1795, full
    ml1796 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1796, ml1795)

    # R: ml1788, full, L: ml1789, full, y: ml1792, full, tmp25: ml1796, full
    ml1797 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1796, 0.0, ml1797)

    # R: ml1788, full, L: ml1789, full, y: ml1792, full, tmp19: ml1797, symmetric_lower_triangular
    ml1798 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1797, ml1792, 0.0, ml1798)

    # R: ml1788, full, L: ml1789, full, tmp19: ml1797, symmetric_lower_triangular, tmp32: ml1798, full
    ml1799 = diag(ml1789)
    ml1800 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1788, 1, ml1800, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1788, 2);
        view(ml1788, :, i)[:] .*= ml1799;
    end;        

    # R: ml1800, full, tmp19: ml1797, symmetric_lower_triangular, tmp32: ml1798, full, tmp29: ml1788, full
    for i = 1:2000-1;
        view(ml1797, i, i+1:2000)[:] = view(ml1797, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1788, ml1800, 1.0, ml1797)

    # tmp32: ml1798, full, tmp31: ml1797, full
    ml1801 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1797, ml1801, info) = LinearAlgebra.LAPACK.getrf!(ml1797)

    # tmp32: ml1798, full, P35: ml1801, ipiv, L33: ml1797, lower_triangular_udiag, U34: ml1797, upper_triangular
    ml1802 = [1:length(ml1801);]
    @inbounds for i in 1:length(ml1801)
        ml1802[i], ml1802[ml1801[i]] = ml1802[ml1801[i]], ml1802[i];
    end;
    ml1803 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1803 = ml1798[ml1802]

    # L33: ml1797, lower_triangular_udiag, U34: ml1797, upper_triangular, tmp40: ml1803, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1797, ml1803)

    # U34: ml1797, upper_triangular, tmp41: ml1803, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1797, ml1803)

    # tmp17: ml1803, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1803), (finish-start)*1e-9)
end