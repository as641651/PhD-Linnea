using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm52(ml1756::Array{Float64,2}, ml1757::Array{Float64,2}, ml1758::Array{Float64,2}, ml1759::Array{Float64,2}, ml1760::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1756, full, L: ml1757, full, A: ml1758, full, B: ml1759, full, y: ml1760, full
    ml1761 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1758, ml1761, info) = LinearAlgebra.LAPACK.getrf!(ml1758)

    # R: ml1756, full, L: ml1757, full, B: ml1759, full, y: ml1760, full, P11: ml1761, ipiv, L9: ml1758, lower_triangular_udiag, U10: ml1758, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1758, ml1759)

    # R: ml1756, full, L: ml1757, full, y: ml1760, full, P11: ml1761, ipiv, L9: ml1758, lower_triangular_udiag, tmp53: ml1759, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1758, ml1759)

    # R: ml1756, full, L: ml1757, full, y: ml1760, full, P11: ml1761, ipiv, tmp54: ml1759, full
    ml1762 = [1:length(ml1761);]
    @inbounds for i in 1:length(ml1761)
        ml1762[i], ml1762[ml1761[i]] = ml1762[ml1761[i]], ml1762[i];
    end;
    ml1763 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1763 = ml1759[:,invperm(ml1762)]

    # R: ml1756, full, L: ml1757, full, y: ml1760, full, tmp55: ml1763, full
    ml1764 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1764, ml1763)

    # R: ml1756, full, L: ml1757, full, y: ml1760, full, tmp25: ml1764, full
    ml1765 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1764, 0.0, ml1765)

    # R: ml1756, full, L: ml1757, full, y: ml1760, full, tmp19: ml1765, symmetric_lower_triangular
    ml1766 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1765, ml1760, 0.0, ml1766)

    # R: ml1756, full, L: ml1757, full, tmp19: ml1765, symmetric_lower_triangular, tmp32: ml1766, full
    ml1767 = diag(ml1757)
    ml1768 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1756, 1, ml1768, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1756, 2);
        view(ml1756, :, i)[:] .*= ml1767;
    end;        

    # R: ml1768, full, tmp19: ml1765, symmetric_lower_triangular, tmp32: ml1766, full, tmp29: ml1756, full
    for i = 1:2000-1;
        view(ml1765, i, i+1:2000)[:] = view(ml1765, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1756, ml1768, 1.0, ml1765)

    # tmp32: ml1766, full, tmp31: ml1765, full
    ml1769 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1765, ml1769, info) = LinearAlgebra.LAPACK.getrf!(ml1765)

    # tmp32: ml1766, full, P35: ml1769, ipiv, L33: ml1765, lower_triangular_udiag, U34: ml1765, upper_triangular
    ml1770 = [1:length(ml1769);]
    @inbounds for i in 1:length(ml1769)
        ml1770[i], ml1770[ml1769[i]] = ml1770[ml1769[i]], ml1770[i];
    end;
    ml1771 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1771 = ml1766[ml1770]

    # L33: ml1765, lower_triangular_udiag, U34: ml1765, upper_triangular, tmp40: ml1771, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1765, ml1771)

    # U34: ml1765, upper_triangular, tmp41: ml1771, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1765, ml1771)

    # tmp17: ml1771, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1771), (finish-start)*1e-9)
end