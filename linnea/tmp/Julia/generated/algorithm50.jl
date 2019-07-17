using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm50(ml1676::Array{Float64,2}, ml1677::Array{Float64,2}, ml1678::Array{Float64,2}, ml1679::Array{Float64,2}, ml1680::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1676, full, L: ml1677, full, A: ml1678, full, B: ml1679, full, y: ml1680, full
    ml1681 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1678, ml1681, info) = LinearAlgebra.LAPACK.getrf!(ml1678)

    # R: ml1676, full, L: ml1677, full, B: ml1679, full, y: ml1680, full, P11: ml1681, ipiv, L9: ml1678, lower_triangular_udiag, U10: ml1678, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1678, ml1679)

    # R: ml1676, full, L: ml1677, full, y: ml1680, full, P11: ml1681, ipiv, L9: ml1678, lower_triangular_udiag, tmp53: ml1679, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1678, ml1679)

    # R: ml1676, full, L: ml1677, full, y: ml1680, full, P11: ml1681, ipiv, tmp54: ml1679, full
    ml1682 = [1:length(ml1681);]
    @inbounds for i in 1:length(ml1681)
        ml1682[i], ml1682[ml1681[i]] = ml1682[ml1681[i]], ml1682[i];
    end;
    ml1683 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1683 = ml1679[:,invperm(ml1682)]

    # R: ml1676, full, L: ml1677, full, y: ml1680, full, tmp55: ml1683, full
    ml1684 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1684, ml1683)

    # R: ml1676, full, L: ml1677, full, y: ml1680, full, tmp25: ml1684, full
    ml1685 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1684, 0.0, ml1685)

    # R: ml1676, full, L: ml1677, full, y: ml1680, full, tmp19: ml1685, symmetric_lower_triangular
    ml1686 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1685, ml1680, 0.0, ml1686)

    # R: ml1676, full, L: ml1677, full, tmp19: ml1685, symmetric_lower_triangular, tmp32: ml1686, full
    ml1687 = diag(ml1677)
    ml1688 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1676, 1, ml1688, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1676, 2);
        view(ml1676, :, i)[:] .*= ml1687;
    end;        

    # R: ml1688, full, tmp19: ml1685, symmetric_lower_triangular, tmp32: ml1686, full, tmp29: ml1676, full
    for i = 1:2000-1;
        view(ml1685, i, i+1:2000)[:] = view(ml1685, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1676, ml1688, 1.0, ml1685)

    # tmp32: ml1686, full, tmp31: ml1685, full
    ml1689 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1685, ml1689, info) = LinearAlgebra.LAPACK.getrf!(ml1685)

    # tmp32: ml1686, full, P35: ml1689, ipiv, L33: ml1685, lower_triangular_udiag, U34: ml1685, upper_triangular
    ml1690 = [1:length(ml1689);]
    @inbounds for i in 1:length(ml1689)
        ml1690[i], ml1690[ml1689[i]] = ml1690[ml1689[i]], ml1690[i];
    end;
    ml1691 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1691 = ml1686[ml1690]

    # L33: ml1685, lower_triangular_udiag, U34: ml1685, upper_triangular, tmp40: ml1691, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1685, ml1691)

    # U34: ml1685, upper_triangular, tmp41: ml1691, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1685, ml1691)

    # tmp17: ml1691, full
    # x = tmp17
    return (ml1691)
end