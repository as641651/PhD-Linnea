using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm47(ml1574::Array{Float64,2}, ml1575::Array{Float64,2}, ml1576::Array{Float64,2}, ml1577::Array{Float64,2}, ml1578::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1574, full, L: ml1575, full, A: ml1576, full, B: ml1577, full, y: ml1578, full
    ml1579 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1579, ml1577)

    # R: ml1574, full, L: ml1575, full, A: ml1576, full, y: ml1578, full, tmp26: ml1579, full
    ml1580 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1576, ml1580, info) = LinearAlgebra.LAPACK.getrf!(ml1576)

    # R: ml1574, full, L: ml1575, full, y: ml1578, full, tmp26: ml1579, full, P11: ml1580, ipiv, L9: ml1576, lower_triangular_udiag, U10: ml1576, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1576, ml1579)

    # R: ml1574, full, L: ml1575, full, y: ml1578, full, P11: ml1580, ipiv, L9: ml1576, lower_triangular_udiag, tmp27: ml1579, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1576, ml1579)

    # R: ml1574, full, L: ml1575, full, y: ml1578, full, P11: ml1580, ipiv, tmp28: ml1579, full
    ml1581 = [1:length(ml1580);]
    @inbounds for i in 1:length(ml1580)
        ml1581[i], ml1581[ml1580[i]] = ml1581[ml1580[i]], ml1581[i];
    end;
    ml1582 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1582 = ml1579[invperm(ml1581),:]

    # R: ml1574, full, L: ml1575, full, y: ml1578, full, tmp25: ml1582, full
    ml1583 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1582, 0.0, ml1583)

    # R: ml1574, full, L: ml1575, full, y: ml1578, full, tmp19: ml1583, symmetric_lower_triangular
    ml1584 = diag(ml1575)
    ml1585 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1574, 1, ml1585, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1574, 2);
        view(ml1574, :, i)[:] .*= ml1584;
    end;        

    # R: ml1585, full, y: ml1578, full, tmp19: ml1583, symmetric_lower_triangular, tmp29: ml1574, full
    for i = 1:2000-1;
        view(ml1583, i, i+1:2000)[:] = view(ml1583, i+1:2000, i);
    end;
    ml1586 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1583, 1, ml1586, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1574, ml1585, 1.0, ml1583)

    # y: ml1578, full, tmp19: ml1586, full, tmp31: ml1583, full
    ml1587 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1583, ml1587, info) = LinearAlgebra.LAPACK.getrf!(ml1583)

    # y: ml1578, full, tmp19: ml1586, full, P35: ml1587, ipiv, L33: ml1583, lower_triangular_udiag, U34: ml1583, upper_triangular
    ml1588 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1586, ml1578, 0.0, ml1588)

    # P35: ml1587, ipiv, L33: ml1583, lower_triangular_udiag, U34: ml1583, upper_triangular, tmp32: ml1588, full
    ml1589 = [1:length(ml1587);]
    @inbounds for i in 1:length(ml1587)
        ml1589[i], ml1589[ml1587[i]] = ml1589[ml1587[i]], ml1589[i];
    end;
    ml1590 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1590 = ml1588[ml1589]

    # L33: ml1583, lower_triangular_udiag, U34: ml1583, upper_triangular, tmp40: ml1590, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1583, ml1590)

    # U34: ml1583, upper_triangular, tmp41: ml1590, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1583, ml1590)

    # tmp17: ml1590, full
    # x = tmp17
    return (ml1590)
end