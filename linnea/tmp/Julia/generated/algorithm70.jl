using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm70(ml2340::Array{Float64,2}, ml2341::Array{Float64,2}, ml2342::Array{Float64,2}, ml2343::Array{Float64,2}, ml2344::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2340, full, L: ml2341, full, A: ml2342, full, B: ml2343, full, y: ml2344, full
    ml2345 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2345, ml2343)

    # R: ml2340, full, L: ml2341, full, A: ml2342, full, y: ml2344, full, tmp26: ml2345, full
    ml2346 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2342, ml2346, info) = LinearAlgebra.LAPACK.getrf!(ml2342)

    # R: ml2340, full, L: ml2341, full, y: ml2344, full, tmp26: ml2345, full, P11: ml2346, ipiv, L9: ml2342, lower_triangular_udiag, U10: ml2342, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2342, ml2345)

    # R: ml2340, full, L: ml2341, full, y: ml2344, full, P11: ml2346, ipiv, L9: ml2342, lower_triangular_udiag, tmp27: ml2345, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2342, ml2345)

    # R: ml2340, full, L: ml2341, full, y: ml2344, full, P11: ml2346, ipiv, tmp28: ml2345, full
    ml2347 = [1:length(ml2346);]
    @inbounds for i in 1:length(ml2346)
        ml2347[i], ml2347[ml2346[i]] = ml2347[ml2346[i]], ml2347[i];
    end;
    ml2348 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2348 = ml2345[invperm(ml2347),:]

    # R: ml2340, full, L: ml2341, full, y: ml2344, full, tmp25: ml2348, full
    ml2349 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2348, 0.0, ml2349)

    # R: ml2340, full, L: ml2341, full, y: ml2344, full, tmp19: ml2349, symmetric_lower_triangular
    ml2350 = diag(ml2341)
    ml2351 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2340, 1, ml2351, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2340, 2);
        view(ml2340, :, i)[:] .*= ml2350;
    end;        

    # R: ml2351, full, y: ml2344, full, tmp19: ml2349, symmetric_lower_triangular, tmp29: ml2340, full
    for i = 1:2000-1;
        view(ml2349, i, i+1:2000)[:] = view(ml2349, i+1:2000, i);
    end;
    ml2352 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2349, 1, ml2352, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2340, ml2351, 1.0, ml2349)

    # y: ml2344, full, tmp19: ml2352, full, tmp31: ml2349, full
    ml2353 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2352, ml2344, 0.0, ml2353)

    # tmp31: ml2349, full, tmp32: ml2353, full
    ml2354 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2349, ml2354, info) = LinearAlgebra.LAPACK.getrf!(ml2349)

    # tmp32: ml2353, full, P35: ml2354, ipiv, L33: ml2349, lower_triangular_udiag, U34: ml2349, upper_triangular
    ml2355 = [1:length(ml2354);]
    @inbounds for i in 1:length(ml2354)
        ml2355[i], ml2355[ml2354[i]] = ml2355[ml2354[i]], ml2355[i];
    end;
    ml2356 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2356 = ml2353[ml2355]

    # L33: ml2349, lower_triangular_udiag, U34: ml2349, upper_triangular, tmp40: ml2356, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2349, ml2356)

    # U34: ml2349, upper_triangular, tmp41: ml2356, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2349, ml2356)

    # tmp17: ml2356, full
    # x = tmp17
    return (ml2356)
end