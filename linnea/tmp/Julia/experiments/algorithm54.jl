using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm54(ml1821::Array{Float64,2}, ml1822::Array{Float64,2}, ml1823::Array{Float64,2}, ml1824::Array{Float64,2}, ml1825::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1821, full, L: ml1822, full, A: ml1823, full, B: ml1824, full, y: ml1825, full
    ml1826 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1826, ml1824)

    # R: ml1821, full, L: ml1822, full, A: ml1823, full, y: ml1825, full, tmp26: ml1826, full
    ml1827 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1823, ml1827, info) = LinearAlgebra.LAPACK.getrf!(ml1823)

    # R: ml1821, full, L: ml1822, full, y: ml1825, full, tmp26: ml1826, full, P11: ml1827, ipiv, L9: ml1823, lower_triangular_udiag, U10: ml1823, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1823, ml1826)

    # R: ml1821, full, L: ml1822, full, y: ml1825, full, P11: ml1827, ipiv, L9: ml1823, lower_triangular_udiag, tmp27: ml1826, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1823, ml1826)

    # R: ml1821, full, L: ml1822, full, y: ml1825, full, P11: ml1827, ipiv, tmp28: ml1826, full
    ml1828 = [1:length(ml1827);]
    @inbounds for i in 1:length(ml1827)
        ml1828[i], ml1828[ml1827[i]] = ml1828[ml1827[i]], ml1828[i];
    end;
    ml1829 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1829 = ml1826[invperm(ml1828),:]

    # R: ml1821, full, L: ml1822, full, y: ml1825, full, tmp25: ml1829, full
    ml1830 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1829, 0.0, ml1830)

    # R: ml1821, full, L: ml1822, full, y: ml1825, full, tmp19: ml1830, symmetric_lower_triangular
    ml1831 = diag(ml1822)
    ml1832 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1821, 1, ml1832, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1821, 2);
        view(ml1821, :, i)[:] .*= ml1831;
    end;        

    # R: ml1832, full, y: ml1825, full, tmp19: ml1830, symmetric_lower_triangular, tmp29: ml1821, full
    for i = 1:2000-1;
        view(ml1830, i, i+1:2000)[:] = view(ml1830, i+1:2000, i);
    end;
    ml1833 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1830, 1, ml1833, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1821, ml1832, 1.0, ml1830)

    # y: ml1825, full, tmp19: ml1833, full, tmp31: ml1830, full
    ml1834 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1830, ml1834, info) = LinearAlgebra.LAPACK.getrf!(ml1830)

    # y: ml1825, full, tmp19: ml1833, full, P35: ml1834, ipiv, L33: ml1830, lower_triangular_udiag, U34: ml1830, upper_triangular
    ml1835 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1833, ml1825, 0.0, ml1835)

    # P35: ml1834, ipiv, L33: ml1830, lower_triangular_udiag, U34: ml1830, upper_triangular, tmp32: ml1835, full
    ml1836 = [1:length(ml1834);]
    @inbounds for i in 1:length(ml1834)
        ml1836[i], ml1836[ml1834[i]] = ml1836[ml1834[i]], ml1836[i];
    end;
    ml1837 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1837 = ml1835[ml1836]

    # L33: ml1830, lower_triangular_udiag, U34: ml1830, upper_triangular, tmp40: ml1837, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1830, ml1837)

    # U34: ml1830, upper_triangular, tmp41: ml1837, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1830, ml1837)

    # tmp17: ml1837, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1837), (finish-start)*1e-9)
end