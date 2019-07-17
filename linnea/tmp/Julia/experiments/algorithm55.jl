using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm55(ml1855::Array{Float64,2}, ml1856::Array{Float64,2}, ml1857::Array{Float64,2}, ml1858::Array{Float64,2}, ml1859::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1855, full, L: ml1856, full, A: ml1857, full, B: ml1858, full, y: ml1859, full
    ml1860 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1857, ml1860, info) = LinearAlgebra.LAPACK.getrf!(ml1857)

    # R: ml1855, full, L: ml1856, full, B: ml1858, full, y: ml1859, full, P11: ml1860, ipiv, L9: ml1857, lower_triangular_udiag, U10: ml1857, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1857, ml1858)

    # R: ml1855, full, L: ml1856, full, y: ml1859, full, P11: ml1860, ipiv, L9: ml1857, lower_triangular_udiag, tmp53: ml1858, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1857, ml1858)

    # R: ml1855, full, L: ml1856, full, y: ml1859, full, P11: ml1860, ipiv, tmp54: ml1858, full
    ml1861 = [1:length(ml1860);]
    @inbounds for i in 1:length(ml1860)
        ml1861[i], ml1861[ml1860[i]] = ml1861[ml1860[i]], ml1861[i];
    end;
    ml1862 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1862 = ml1858[:,invperm(ml1861)]

    # R: ml1855, full, L: ml1856, full, y: ml1859, full, tmp55: ml1862, full
    ml1863 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1863, ml1862)

    # R: ml1855, full, L: ml1856, full, y: ml1859, full, tmp25: ml1863, full
    ml1864 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1863, 0.0, ml1864)

    # R: ml1855, full, L: ml1856, full, y: ml1859, full, tmp19: ml1864, symmetric_lower_triangular
    ml1865 = diag(ml1856)
    ml1866 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1855, 1, ml1866, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1855, 2);
        view(ml1855, :, i)[:] .*= ml1865;
    end;        

    # R: ml1866, full, y: ml1859, full, tmp19: ml1864, symmetric_lower_triangular, tmp29: ml1855, full
    for i = 1:2000-1;
        view(ml1864, i, i+1:2000)[:] = view(ml1864, i+1:2000, i);
    end;
    ml1867 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1864, 1, ml1867, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1855, ml1866, 1.0, ml1864)

    # y: ml1859, full, tmp19: ml1867, full, tmp31: ml1864, full
    ml1868 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1864, ml1868, info) = LinearAlgebra.LAPACK.getrf!(ml1864)

    # y: ml1859, full, tmp19: ml1867, full, P35: ml1868, ipiv, L33: ml1864, lower_triangular_udiag, U34: ml1864, upper_triangular
    ml1869 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1867, ml1859, 0.0, ml1869)

    # P35: ml1868, ipiv, L33: ml1864, lower_triangular_udiag, U34: ml1864, upper_triangular, tmp32: ml1869, full
    ml1870 = [1:length(ml1868);]
    @inbounds for i in 1:length(ml1868)
        ml1870[i], ml1870[ml1868[i]] = ml1870[ml1868[i]], ml1870[i];
    end;
    ml1871 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1871 = ml1869[ml1870]

    # L33: ml1864, lower_triangular_udiag, U34: ml1864, upper_triangular, tmp40: ml1871, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1864, ml1871)

    # U34: ml1864, upper_triangular, tmp41: ml1871, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1864, ml1871)

    # tmp17: ml1871, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1871), (finish-start)*1e-9)
end