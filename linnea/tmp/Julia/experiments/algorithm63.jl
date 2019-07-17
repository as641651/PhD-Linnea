using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm63(ml2120::Array{Float64,2}, ml2121::Array{Float64,2}, ml2122::Array{Float64,2}, ml2123::Array{Float64,2}, ml2124::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2120, full, L: ml2121, full, A: ml2122, full, B: ml2123, full, y: ml2124, full
    ml2125 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2125, ml2123)

    # R: ml2120, full, L: ml2121, full, A: ml2122, full, y: ml2124, full, tmp26: ml2125, full
    ml2126 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2122, ml2126, info) = LinearAlgebra.LAPACK.getrf!(ml2122)

    # R: ml2120, full, L: ml2121, full, y: ml2124, full, tmp26: ml2125, full, P11: ml2126, ipiv, L9: ml2122, lower_triangular_udiag, U10: ml2122, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2122, ml2125)

    # R: ml2120, full, L: ml2121, full, y: ml2124, full, P11: ml2126, ipiv, L9: ml2122, lower_triangular_udiag, tmp27: ml2125, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2122, ml2125)

    # R: ml2120, full, L: ml2121, full, y: ml2124, full, P11: ml2126, ipiv, tmp28: ml2125, full
    ml2127 = [1:length(ml2126);]
    @inbounds for i in 1:length(ml2126)
        ml2127[i], ml2127[ml2126[i]] = ml2127[ml2126[i]], ml2127[i];
    end;
    ml2128 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2128 = ml2125[invperm(ml2127),:]

    # R: ml2120, full, L: ml2121, full, y: ml2124, full, tmp25: ml2128, full
    ml2129 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2128, 0.0, ml2129)

    # R: ml2120, full, L: ml2121, full, y: ml2124, full, tmp19: ml2129, symmetric_lower_triangular
    ml2130 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2129, ml2124, 0.0, ml2130)

    # R: ml2120, full, L: ml2121, full, tmp19: ml2129, symmetric_lower_triangular, tmp32: ml2130, full
    ml2131 = diag(ml2121)
    ml2132 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2120, 1, ml2132, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2120, 2);
        view(ml2120, :, i)[:] .*= ml2131;
    end;        

    # R: ml2132, full, tmp19: ml2129, symmetric_lower_triangular, tmp32: ml2130, full, tmp29: ml2120, full
    for i = 1:2000-1;
        view(ml2129, i, i+1:2000)[:] = view(ml2129, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2132, ml2120, 1.0, ml2129)

    # tmp32: ml2130, full, tmp31: ml2129, full
    ml2133 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2129, ml2133, info) = LinearAlgebra.LAPACK.getrf!(ml2129)

    # tmp32: ml2130, full, P35: ml2133, ipiv, L33: ml2129, lower_triangular_udiag, U34: ml2129, upper_triangular
    ml2134 = [1:length(ml2133);]
    @inbounds for i in 1:length(ml2133)
        ml2134[i], ml2134[ml2133[i]] = ml2134[ml2133[i]], ml2134[i];
    end;
    ml2135 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2135 = ml2130[ml2134]

    # L33: ml2129, lower_triangular_udiag, U34: ml2129, upper_triangular, tmp40: ml2135, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2129, ml2135)

    # U34: ml2129, upper_triangular, tmp41: ml2135, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2129, ml2135)

    # tmp17: ml2135, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2135), (finish-start)*1e-9)
end