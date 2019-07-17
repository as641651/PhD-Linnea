using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm51(ml1708::Array{Float64,2}, ml1709::Array{Float64,2}, ml1710::Array{Float64,2}, ml1711::Array{Float64,2}, ml1712::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1708, full, L: ml1709, full, A: ml1710, full, B: ml1711, full, y: ml1712, full
    ml1713 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1710, ml1713, info) = LinearAlgebra.LAPACK.getrf!(ml1710)

    # R: ml1708, full, L: ml1709, full, B: ml1711, full, y: ml1712, full, P11: ml1713, ipiv, L9: ml1710, lower_triangular_udiag, U10: ml1710, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1710, ml1711)

    # R: ml1708, full, L: ml1709, full, y: ml1712, full, P11: ml1713, ipiv, L9: ml1710, lower_triangular_udiag, tmp53: ml1711, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1710, ml1711)

    # R: ml1708, full, L: ml1709, full, y: ml1712, full, P11: ml1713, ipiv, tmp54: ml1711, full
    ml1714 = [1:length(ml1713);]
    @inbounds for i in 1:length(ml1713)
        ml1714[i], ml1714[ml1713[i]] = ml1714[ml1713[i]], ml1714[i];
    end;
    ml1715 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1715 = ml1711[:,invperm(ml1714)]

    # R: ml1708, full, L: ml1709, full, y: ml1712, full, tmp55: ml1715, full
    ml1716 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1716, ml1715)

    # R: ml1708, full, L: ml1709, full, y: ml1712, full, tmp25: ml1716, full
    ml1717 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1716, 0.0, ml1717)

    # R: ml1708, full, L: ml1709, full, y: ml1712, full, tmp19: ml1717, symmetric_lower_triangular
    ml1718 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1717, ml1712, 0.0, ml1718)

    # R: ml1708, full, L: ml1709, full, tmp19: ml1717, symmetric_lower_triangular, tmp32: ml1718, full
    ml1719 = diag(ml1709)
    ml1720 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1708, 1, ml1720, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1708, 2);
        view(ml1708, :, i)[:] .*= ml1719;
    end;        

    # R: ml1720, full, tmp19: ml1717, symmetric_lower_triangular, tmp32: ml1718, full, tmp29: ml1708, full
    for i = 1:2000-1;
        view(ml1717, i, i+1:2000)[:] = view(ml1717, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1708, ml1720, 1.0, ml1717)

    # tmp32: ml1718, full, tmp31: ml1717, full
    ml1721 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1717, ml1721, info) = LinearAlgebra.LAPACK.getrf!(ml1717)

    # tmp32: ml1718, full, P35: ml1721, ipiv, L33: ml1717, lower_triangular_udiag, U34: ml1717, upper_triangular
    ml1722 = [1:length(ml1721);]
    @inbounds for i in 1:length(ml1721)
        ml1722[i], ml1722[ml1721[i]] = ml1722[ml1721[i]], ml1722[i];
    end;
    ml1723 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1723 = ml1718[ml1722]

    # L33: ml1717, lower_triangular_udiag, U34: ml1717, upper_triangular, tmp40: ml1723, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1717, ml1723)

    # U34: ml1717, upper_triangular, tmp41: ml1723, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1717, ml1723)

    # tmp17: ml1723, full
    # x = tmp17
    return (ml1723)
end