using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm89(ml2995::Array{Float64,2}, ml2996::Array{Float64,2}, ml2997::Array{Float64,2}, ml2998::Array{Float64,2}, ml2999::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2995, full, L: ml2996, full, A: ml2997, full, B: ml2998, full, y: ml2999, full
    ml3000 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2997, ml3000, info) = LinearAlgebra.LAPACK.getrf!(ml2997)

    # R: ml2995, full, L: ml2996, full, B: ml2998, full, y: ml2999, full, P11: ml3000, ipiv, L9: ml2997, lower_triangular_udiag, U10: ml2997, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml2997, ml2998)

    # R: ml2995, full, L: ml2996, full, y: ml2999, full, P11: ml3000, ipiv, L9: ml2997, lower_triangular_udiag, tmp53: ml2998, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml2997, ml2998)

    # R: ml2995, full, L: ml2996, full, y: ml2999, full, P11: ml3000, ipiv, tmp54: ml2998, full
    ml3001 = [1:length(ml3000);]
    @inbounds for i in 1:length(ml3000)
        ml3001[i], ml3001[ml3000[i]] = ml3001[ml3000[i]], ml3001[i];
    end;
    ml3002 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml3002 = ml2998[:,invperm(ml3001)]

    # R: ml2995, full, L: ml2996, full, y: ml2999, full, tmp55: ml3002, full
    ml3003 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml3003, ml3002)

    # R: ml2995, full, L: ml2996, full, y: ml2999, full, tmp25: ml3003, full
    ml3004 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3003, 0.0, ml3004)

    # R: ml2995, full, L: ml2996, full, y: ml2999, full, tmp19: ml3004, symmetric_lower_triangular
    ml3005 = diag(ml2996)
    ml3006 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2995, 1, ml3006, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2995, 2);
        view(ml2995, :, i)[:] .*= ml3005;
    end;        

    # R: ml3006, full, y: ml2999, full, tmp19: ml3004, symmetric_lower_triangular, tmp29: ml2995, full
    for i = 1:2000-1;
        view(ml3004, i, i+1:2000)[:] = view(ml3004, i+1:2000, i);
    end;
    ml3007 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3004, 1, ml3007, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2995, ml3006, 1.0, ml3004)

    # y: ml2999, full, tmp19: ml3007, full, tmp31: ml3004, full
    ml3008 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3004, ml3008, info) = LinearAlgebra.LAPACK.getrf!(ml3004)

    # y: ml2999, full, tmp19: ml3007, full, P35: ml3008, ipiv, L33: ml3004, lower_triangular_udiag, U34: ml3004, upper_triangular
    ml3009 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3007, ml2999, 0.0, ml3009)

    # P35: ml3008, ipiv, L33: ml3004, lower_triangular_udiag, U34: ml3004, upper_triangular, tmp32: ml3009, full
    ml3010 = [1:length(ml3008);]
    @inbounds for i in 1:length(ml3008)
        ml3010[i], ml3010[ml3008[i]] = ml3010[ml3008[i]], ml3010[i];
    end;
    ml3011 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3011 = ml3009[ml3010]

    # L33: ml3004, lower_triangular_udiag, U34: ml3004, upper_triangular, tmp40: ml3011, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3004, ml3011)

    # U34: ml3004, upper_triangular, tmp41: ml3011, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3004, ml3011)

    # tmp17: ml3011, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3011), (finish-start)*1e-9)
end