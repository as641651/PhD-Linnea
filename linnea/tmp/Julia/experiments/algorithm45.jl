using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm45(ml1523::Array{Float64,2}, ml1524::Array{Float64,2}, ml1525::Array{Float64,2}, ml1526::Array{Float64,2}, ml1527::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1523, full, L: ml1524, full, A: ml1525, full, B: ml1526, full, y: ml1527, full
    ml1528 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1528, ml1526)

    # R: ml1523, full, L: ml1524, full, A: ml1525, full, y: ml1527, full, tmp26: ml1528, full
    ml1529 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1525, ml1529, info) = LinearAlgebra.LAPACK.getrf!(ml1525)

    # R: ml1523, full, L: ml1524, full, y: ml1527, full, tmp26: ml1528, full, P11: ml1529, ipiv, L9: ml1525, lower_triangular_udiag, U10: ml1525, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1525, ml1528)

    # R: ml1523, full, L: ml1524, full, y: ml1527, full, P11: ml1529, ipiv, L9: ml1525, lower_triangular_udiag, tmp27: ml1528, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1525, ml1528)

    # R: ml1523, full, L: ml1524, full, y: ml1527, full, P11: ml1529, ipiv, tmp28: ml1528, full
    ml1530 = [1:length(ml1529);]
    @inbounds for i in 1:length(ml1529)
        ml1530[i], ml1530[ml1529[i]] = ml1530[ml1529[i]], ml1530[i];
    end;
    ml1531 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1531 = ml1528[invperm(ml1530),:]

    # R: ml1523, full, L: ml1524, full, y: ml1527, full, tmp25: ml1531, full
    ml1532 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1531, 0.0, ml1532)

    # R: ml1523, full, L: ml1524, full, y: ml1527, full, tmp19: ml1532, symmetric_lower_triangular
    ml1533 = diag(ml1524)
    ml1534 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1523, 1, ml1534, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1523, 2);
        view(ml1523, :, i)[:] .*= ml1533;
    end;        

    # R: ml1534, full, y: ml1527, full, tmp19: ml1532, symmetric_lower_triangular, tmp29: ml1523, full
    for i = 1:2000-1;
        view(ml1532, i, i+1:2000)[:] = view(ml1532, i+1:2000, i);
    end;
    ml1535 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1532, 1, ml1535, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1523, ml1534, 1.0, ml1532)

    # y: ml1527, full, tmp19: ml1535, full, tmp31: ml1532, full
    ml1536 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1532, ml1536, info) = LinearAlgebra.LAPACK.getrf!(ml1532)

    # y: ml1527, full, tmp19: ml1535, full, P35: ml1536, ipiv, L33: ml1532, lower_triangular_udiag, U34: ml1532, upper_triangular
    ml1537 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1535, ml1527, 0.0, ml1537)

    # P35: ml1536, ipiv, L33: ml1532, lower_triangular_udiag, U34: ml1532, upper_triangular, tmp32: ml1537, full
    ml1538 = [1:length(ml1536);]
    @inbounds for i in 1:length(ml1536)
        ml1538[i], ml1538[ml1536[i]] = ml1538[ml1536[i]], ml1538[i];
    end;
    ml1539 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1539 = ml1537[ml1538]

    # L33: ml1532, lower_triangular_udiag, U34: ml1532, upper_triangular, tmp40: ml1539, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1532, ml1539)

    # U34: ml1532, upper_triangular, tmp41: ml1539, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1532, ml1539)

    # tmp17: ml1539, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1539), (finish-start)*1e-9)
end