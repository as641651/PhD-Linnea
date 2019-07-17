using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm43(ml1455::Array{Float64,2}, ml1456::Array{Float64,2}, ml1457::Array{Float64,2}, ml1458::Array{Float64,2}, ml1459::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1455, full, L: ml1456, full, A: ml1457, full, B: ml1458, full, y: ml1459, full
    ml1460 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1460, ml1458)

    # R: ml1455, full, L: ml1456, full, A: ml1457, full, y: ml1459, full, tmp26: ml1460, full
    ml1461 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1457, ml1461, info) = LinearAlgebra.LAPACK.getrf!(ml1457)

    # R: ml1455, full, L: ml1456, full, y: ml1459, full, tmp26: ml1460, full, P11: ml1461, ipiv, L9: ml1457, lower_triangular_udiag, U10: ml1457, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1457, ml1460)

    # R: ml1455, full, L: ml1456, full, y: ml1459, full, P11: ml1461, ipiv, L9: ml1457, lower_triangular_udiag, tmp27: ml1460, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1457, ml1460)

    # R: ml1455, full, L: ml1456, full, y: ml1459, full, P11: ml1461, ipiv, tmp28: ml1460, full
    ml1462 = [1:length(ml1461);]
    @inbounds for i in 1:length(ml1461)
        ml1462[i], ml1462[ml1461[i]] = ml1462[ml1461[i]], ml1462[i];
    end;
    ml1463 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1463 = ml1460[invperm(ml1462),:]

    # R: ml1455, full, L: ml1456, full, y: ml1459, full, tmp25: ml1463, full
    ml1464 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1463, 0.0, ml1464)

    # R: ml1455, full, L: ml1456, full, y: ml1459, full, tmp19: ml1464, symmetric_lower_triangular
    ml1465 = diag(ml1456)
    ml1466 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1455, 1, ml1466, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1455, 2);
        view(ml1455, :, i)[:] .*= ml1465;
    end;        

    # R: ml1466, full, y: ml1459, full, tmp19: ml1464, symmetric_lower_triangular, tmp29: ml1455, full
    for i = 1:2000-1;
        view(ml1464, i, i+1:2000)[:] = view(ml1464, i+1:2000, i);
    end;
    ml1467 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1464, 1, ml1467, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1455, ml1466, 1.0, ml1464)

    # y: ml1459, full, tmp19: ml1467, full, tmp31: ml1464, full
    ml1468 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1467, ml1459, 0.0, ml1468)

    # tmp31: ml1464, full, tmp32: ml1468, full
    ml1469 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1464, ml1469, info) = LinearAlgebra.LAPACK.getrf!(ml1464)

    # tmp32: ml1468, full, P35: ml1469, ipiv, L33: ml1464, lower_triangular_udiag, U34: ml1464, upper_triangular
    ml1470 = [1:length(ml1469);]
    @inbounds for i in 1:length(ml1469)
        ml1470[i], ml1470[ml1469[i]] = ml1470[ml1469[i]], ml1470[i];
    end;
    ml1471 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1471 = ml1468[ml1470]

    # L33: ml1464, lower_triangular_udiag, U34: ml1464, upper_triangular, tmp40: ml1471, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1464, ml1471)

    # U34: ml1464, upper_triangular, tmp41: ml1471, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1464, ml1471)

    # tmp17: ml1471, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1471), (finish-start)*1e-9)
end