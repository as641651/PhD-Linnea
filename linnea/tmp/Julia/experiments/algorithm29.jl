using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm29(ml987::Array{Float64,2}, ml988::Array{Float64,2}, ml989::Array{Float64,2}, ml990::Array{Float64,2}, ml991::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml987, full, L: ml988, full, A: ml989, full, B: ml990, full, y: ml991, full
    ml992 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml989, ml992, info) = LinearAlgebra.LAPACK.getrf!(ml989)

    # R: ml987, full, L: ml988, full, B: ml990, full, y: ml991, full, P11: ml992, ipiv, L9: ml989, lower_triangular_udiag, U10: ml989, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml989, ml990)

    # R: ml987, full, L: ml988, full, y: ml991, full, P11: ml992, ipiv, L9: ml989, lower_triangular_udiag, tmp53: ml990, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml989, ml990)

    # R: ml987, full, L: ml988, full, y: ml991, full, P11: ml992, ipiv, tmp54: ml990, full
    ml993 = [1:length(ml992);]
    @inbounds for i in 1:length(ml992)
        ml993[i], ml993[ml992[i]] = ml993[ml992[i]], ml993[i];
    end;
    ml994 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml994 = ml990[:,invperm(ml993)]

    # R: ml987, full, L: ml988, full, y: ml991, full, tmp55: ml994, full
    ml995 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml995, ml994)

    # R: ml987, full, L: ml988, full, y: ml991, full, tmp25: ml995, full
    ml996 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml995, 0.0, ml996)

    # R: ml987, full, L: ml988, full, y: ml991, full, tmp19: ml996, symmetric_lower_triangular
    ml997 = diag(ml988)
    ml998 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml987, 1, ml998, 1)
    # tmp29 = (L R)
    for i = 1:size(ml987, 2);
        view(ml987, :, i)[:] .*= ml997;
    end;        

    # R: ml998, full, y: ml991, full, tmp19: ml996, symmetric_lower_triangular, tmp29: ml987, full
    for i = 1:2000-1;
        view(ml996, i, i+1:2000)[:] = view(ml996, i+1:2000, i);
    end;
    ml999 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml996, 1, ml999, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml987, ml998, 1.0, ml996)

    # y: ml991, full, tmp19: ml999, full, tmp31: ml996, full
    ml1000 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml996, ml1000, info) = LinearAlgebra.LAPACK.getrf!(ml996)

    # y: ml991, full, tmp19: ml999, full, P35: ml1000, ipiv, L33: ml996, lower_triangular_udiag, U34: ml996, upper_triangular
    ml1001 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml999, ml991, 0.0, ml1001)

    # P35: ml1000, ipiv, L33: ml996, lower_triangular_udiag, U34: ml996, upper_triangular, tmp32: ml1001, full
    ml1002 = [1:length(ml1000);]
    @inbounds for i in 1:length(ml1000)
        ml1002[i], ml1002[ml1000[i]] = ml1002[ml1000[i]], ml1002[i];
    end;
    ml1003 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1003 = ml1001[ml1002]

    # L33: ml996, lower_triangular_udiag, U34: ml996, upper_triangular, tmp40: ml1003, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml996, ml1003)

    # U34: ml996, upper_triangular, tmp41: ml1003, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml996, ml1003)

    # tmp17: ml1003, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1003), (finish-start)*1e-9)
end