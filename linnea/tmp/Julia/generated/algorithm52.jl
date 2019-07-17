using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm52(ml1740::Array{Float64,2}, ml1741::Array{Float64,2}, ml1742::Array{Float64,2}, ml1743::Array{Float64,2}, ml1744::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1740, full, L: ml1741, full, A: ml1742, full, B: ml1743, full, y: ml1744, full
    ml1745 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1742, ml1745, info) = LinearAlgebra.LAPACK.getrf!(ml1742)

    # R: ml1740, full, L: ml1741, full, B: ml1743, full, y: ml1744, full, P11: ml1745, ipiv, L9: ml1742, lower_triangular_udiag, U10: ml1742, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1742, ml1743)

    # R: ml1740, full, L: ml1741, full, y: ml1744, full, P11: ml1745, ipiv, L9: ml1742, lower_triangular_udiag, tmp53: ml1743, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1742, ml1743)

    # R: ml1740, full, L: ml1741, full, y: ml1744, full, P11: ml1745, ipiv, tmp54: ml1743, full
    ml1746 = [1:length(ml1745);]
    @inbounds for i in 1:length(ml1745)
        ml1746[i], ml1746[ml1745[i]] = ml1746[ml1745[i]], ml1746[i];
    end;
    ml1747 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1747 = ml1743[:,invperm(ml1746)]

    # R: ml1740, full, L: ml1741, full, y: ml1744, full, tmp55: ml1747, full
    ml1748 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1748, ml1747)

    # R: ml1740, full, L: ml1741, full, y: ml1744, full, tmp25: ml1748, full
    ml1749 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1748, 0.0, ml1749)

    # R: ml1740, full, L: ml1741, full, y: ml1744, full, tmp19: ml1749, symmetric_lower_triangular
    ml1750 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1749, ml1744, 0.0, ml1750)

    # R: ml1740, full, L: ml1741, full, tmp19: ml1749, symmetric_lower_triangular, tmp32: ml1750, full
    ml1751 = diag(ml1741)
    ml1752 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1740, 1, ml1752, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1740, 2);
        view(ml1740, :, i)[:] .*= ml1751;
    end;        

    # R: ml1752, full, tmp19: ml1749, symmetric_lower_triangular, tmp32: ml1750, full, tmp29: ml1740, full
    for i = 1:2000-1;
        view(ml1749, i, i+1:2000)[:] = view(ml1749, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1740, ml1752, 1.0, ml1749)

    # tmp32: ml1750, full, tmp31: ml1749, full
    ml1753 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1749, ml1753, info) = LinearAlgebra.LAPACK.getrf!(ml1749)

    # tmp32: ml1750, full, P35: ml1753, ipiv, L33: ml1749, lower_triangular_udiag, U34: ml1749, upper_triangular
    ml1754 = [1:length(ml1753);]
    @inbounds for i in 1:length(ml1753)
        ml1754[i], ml1754[ml1753[i]] = ml1754[ml1753[i]], ml1754[i];
    end;
    ml1755 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1755 = ml1750[ml1754]

    # L33: ml1749, lower_triangular_udiag, U34: ml1749, upper_triangular, tmp40: ml1755, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1749, ml1755)

    # U34: ml1749, upper_triangular, tmp41: ml1755, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1749, ml1755)

    # tmp17: ml1755, full
    # x = tmp17
    return (ml1755)
end