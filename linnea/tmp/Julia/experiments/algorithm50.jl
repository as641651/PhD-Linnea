using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm50(ml1692::Array{Float64,2}, ml1693::Array{Float64,2}, ml1694::Array{Float64,2}, ml1695::Array{Float64,2}, ml1696::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1692, full, L: ml1693, full, A: ml1694, full, B: ml1695, full, y: ml1696, full
    ml1697 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1694, ml1697, info) = LinearAlgebra.LAPACK.getrf!(ml1694)

    # R: ml1692, full, L: ml1693, full, B: ml1695, full, y: ml1696, full, P11: ml1697, ipiv, L9: ml1694, lower_triangular_udiag, U10: ml1694, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1694, ml1695)

    # R: ml1692, full, L: ml1693, full, y: ml1696, full, P11: ml1697, ipiv, L9: ml1694, lower_triangular_udiag, tmp53: ml1695, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1694, ml1695)

    # R: ml1692, full, L: ml1693, full, y: ml1696, full, P11: ml1697, ipiv, tmp54: ml1695, full
    ml1698 = [1:length(ml1697);]
    @inbounds for i in 1:length(ml1697)
        ml1698[i], ml1698[ml1697[i]] = ml1698[ml1697[i]], ml1698[i];
    end;
    ml1699 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1699 = ml1695[:,invperm(ml1698)]

    # R: ml1692, full, L: ml1693, full, y: ml1696, full, tmp55: ml1699, full
    ml1700 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1700, ml1699)

    # R: ml1692, full, L: ml1693, full, y: ml1696, full, tmp25: ml1700, full
    ml1701 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1700, 0.0, ml1701)

    # R: ml1692, full, L: ml1693, full, y: ml1696, full, tmp19: ml1701, symmetric_lower_triangular
    ml1702 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1701, ml1696, 0.0, ml1702)

    # R: ml1692, full, L: ml1693, full, tmp19: ml1701, symmetric_lower_triangular, tmp32: ml1702, full
    ml1703 = diag(ml1693)
    ml1704 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1692, 1, ml1704, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1692, 2);
        view(ml1692, :, i)[:] .*= ml1703;
    end;        

    # R: ml1704, full, tmp19: ml1701, symmetric_lower_triangular, tmp32: ml1702, full, tmp29: ml1692, full
    for i = 1:2000-1;
        view(ml1701, i, i+1:2000)[:] = view(ml1701, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1692, ml1704, 1.0, ml1701)

    # tmp32: ml1702, full, tmp31: ml1701, full
    ml1705 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1701, ml1705, info) = LinearAlgebra.LAPACK.getrf!(ml1701)

    # tmp32: ml1702, full, P35: ml1705, ipiv, L33: ml1701, lower_triangular_udiag, U34: ml1701, upper_triangular
    ml1706 = [1:length(ml1705);]
    @inbounds for i in 1:length(ml1705)
        ml1706[i], ml1706[ml1705[i]] = ml1706[ml1705[i]], ml1706[i];
    end;
    ml1707 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1707 = ml1702[ml1706]

    # L33: ml1701, lower_triangular_udiag, U34: ml1701, upper_triangular, tmp40: ml1707, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1701, ml1707)

    # U34: ml1701, upper_triangular, tmp41: ml1707, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1701, ml1707)

    # tmp17: ml1707, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1707), (finish-start)*1e-9)
end