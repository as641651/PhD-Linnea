using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm64(ml2153::Array{Float64,2}, ml2154::Array{Float64,2}, ml2155::Array{Float64,2}, ml2156::Array{Float64,2}, ml2157::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2153, full, L: ml2154, full, A: ml2155, full, B: ml2156, full, y: ml2157, full
    ml2158 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2158, ml2156)

    # R: ml2153, full, L: ml2154, full, A: ml2155, full, y: ml2157, full, tmp26: ml2158, full
    ml2159 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2155, ml2159, info) = LinearAlgebra.LAPACK.getrf!(ml2155)

    # R: ml2153, full, L: ml2154, full, y: ml2157, full, tmp26: ml2158, full, P11: ml2159, ipiv, L9: ml2155, lower_triangular_udiag, U10: ml2155, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2155, ml2158)

    # R: ml2153, full, L: ml2154, full, y: ml2157, full, P11: ml2159, ipiv, L9: ml2155, lower_triangular_udiag, tmp27: ml2158, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2155, ml2158)

    # R: ml2153, full, L: ml2154, full, y: ml2157, full, P11: ml2159, ipiv, tmp28: ml2158, full
    ml2160 = [1:length(ml2159);]
    @inbounds for i in 1:length(ml2159)
        ml2160[i], ml2160[ml2159[i]] = ml2160[ml2159[i]], ml2160[i];
    end;
    ml2161 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2161 = ml2158[invperm(ml2160),:]

    # R: ml2153, full, L: ml2154, full, y: ml2157, full, tmp25: ml2161, full
    ml2162 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2161, 0.0, ml2162)

    # R: ml2153, full, L: ml2154, full, y: ml2157, full, tmp19: ml2162, symmetric_lower_triangular
    ml2163 = diag(ml2154)
    ml2164 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2153, 1, ml2164, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2153, 2);
        view(ml2153, :, i)[:] .*= ml2163;
    end;        

    # R: ml2164, full, y: ml2157, full, tmp19: ml2162, symmetric_lower_triangular, tmp29: ml2153, full
    for i = 1:2000-1;
        view(ml2162, i, i+1:2000)[:] = view(ml2162, i+1:2000, i);
    end;
    ml2165 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2162, 1, ml2165, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2153, ml2164, 1.0, ml2162)

    # y: ml2157, full, tmp19: ml2165, full, tmp31: ml2162, full
    ml2166 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2162, ml2166, info) = LinearAlgebra.LAPACK.getrf!(ml2162)

    # y: ml2157, full, tmp19: ml2165, full, P35: ml2166, ipiv, L33: ml2162, lower_triangular_udiag, U34: ml2162, upper_triangular
    ml2167 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2165, ml2157, 0.0, ml2167)

    # P35: ml2166, ipiv, L33: ml2162, lower_triangular_udiag, U34: ml2162, upper_triangular, tmp32: ml2167, full
    ml2168 = [1:length(ml2166);]
    @inbounds for i in 1:length(ml2166)
        ml2168[i], ml2168[ml2166[i]] = ml2168[ml2166[i]], ml2168[i];
    end;
    ml2169 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2169 = ml2167[ml2168]

    # L33: ml2162, lower_triangular_udiag, U34: ml2162, upper_triangular, tmp40: ml2169, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2162, ml2169)

    # U34: ml2162, upper_triangular, tmp41: ml2169, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2162, ml2169)

    # tmp17: ml2169, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2169), (finish-start)*1e-9)
end