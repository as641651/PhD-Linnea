using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm13(ml451::Array{Float64,2}, ml452::Array{Float64,2}, ml453::Array{Float64,2}, ml454::Array{Float64,2}, ml455::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml451, full, L: ml452, full, A: ml453, full, B: ml454, full, y: ml455, full
    ml456 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml453, ml456, info) = LinearAlgebra.LAPACK.getrf!(ml453)

    # R: ml451, full, L: ml452, full, B: ml454, full, y: ml455, full, P11: ml456, ipiv, L9: ml453, lower_triangular_udiag, U10: ml453, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml453, ml454)

    # R: ml451, full, L: ml452, full, y: ml455, full, P11: ml456, ipiv, L9: ml453, lower_triangular_udiag, tmp53: ml454, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml453, ml454)

    # R: ml451, full, L: ml452, full, y: ml455, full, P11: ml456, ipiv, tmp54: ml454, full
    ml457 = [1:length(ml456);]
    @inbounds for i in 1:length(ml456)
        ml457[i], ml457[ml456[i]] = ml457[ml456[i]], ml457[i];
    end;
    ml458 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml458 = ml454[:,invperm(ml457)]

    # R: ml451, full, L: ml452, full, y: ml455, full, tmp55: ml458, full
    ml459 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml459, ml458)

    # R: ml451, full, L: ml452, full, y: ml455, full, tmp25: ml459, full
    ml460 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml459, 0.0, ml460)

    # R: ml451, full, L: ml452, full, y: ml455, full, tmp19: ml460, symmetric_lower_triangular
    ml461 = diag(ml452)
    ml462 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml451, 1, ml462, 1)
    # tmp29 = (L R)
    for i = 1:size(ml451, 2);
        view(ml451, :, i)[:] .*= ml461;
    end;        

    # R: ml462, full, y: ml455, full, tmp19: ml460, symmetric_lower_triangular, tmp29: ml451, full
    for i = 1:2000-1;
        view(ml460, i, i+1:2000)[:] = view(ml460, i+1:2000, i);
    end;
    ml463 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml460, 1, ml463, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml451, ml462, 1.0, ml460)

    # y: ml455, full, tmp19: ml463, full, tmp31: ml460, full
    ml464 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml463, ml455, 0.0, ml464)

    # tmp31: ml460, full, tmp32: ml464, full
    ml465 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml460, ml465, info) = LinearAlgebra.LAPACK.getrf!(ml460)

    # tmp32: ml464, full, P35: ml465, ipiv, L33: ml460, lower_triangular_udiag, U34: ml460, upper_triangular
    ml466 = [1:length(ml465);]
    @inbounds for i in 1:length(ml465)
        ml466[i], ml466[ml465[i]] = ml466[ml465[i]], ml466[i];
    end;
    ml467 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml467 = ml464[ml466]

    # L33: ml460, lower_triangular_udiag, U34: ml460, upper_triangular, tmp40: ml467, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml460, ml467)

    # U34: ml460, upper_triangular, tmp41: ml467, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml460, ml467)

    # tmp17: ml467, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml467), (finish-start)*1e-9)
end