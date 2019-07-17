using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm61(ml2040::Array{Float64,2}, ml2041::Array{Float64,2}, ml2042::Array{Float64,2}, ml2043::Array{Float64,2}, ml2044::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2040, full, L: ml2041, full, A: ml2042, full, B: ml2043, full, y: ml2044, full
    ml2045 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2045, ml2043)

    # R: ml2040, full, L: ml2041, full, A: ml2042, full, y: ml2044, full, tmp26: ml2045, full
    ml2046 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2042, ml2046, info) = LinearAlgebra.LAPACK.getrf!(ml2042)

    # R: ml2040, full, L: ml2041, full, y: ml2044, full, tmp26: ml2045, full, P11: ml2046, ipiv, L9: ml2042, lower_triangular_udiag, U10: ml2042, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2042, ml2045)

    # R: ml2040, full, L: ml2041, full, y: ml2044, full, P11: ml2046, ipiv, L9: ml2042, lower_triangular_udiag, tmp27: ml2045, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2042, ml2045)

    # R: ml2040, full, L: ml2041, full, y: ml2044, full, P11: ml2046, ipiv, tmp28: ml2045, full
    ml2047 = [1:length(ml2046);]
    @inbounds for i in 1:length(ml2046)
        ml2047[i], ml2047[ml2046[i]] = ml2047[ml2046[i]], ml2047[i];
    end;
    ml2048 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2048 = ml2045[invperm(ml2047),:]

    # R: ml2040, full, L: ml2041, full, y: ml2044, full, tmp25: ml2048, full
    ml2049 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2048, 0.0, ml2049)

    # R: ml2040, full, L: ml2041, full, y: ml2044, full, tmp19: ml2049, symmetric_lower_triangular
    ml2050 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2049, ml2044, 0.0, ml2050)

    # R: ml2040, full, L: ml2041, full, tmp19: ml2049, symmetric_lower_triangular, tmp32: ml2050, full
    ml2051 = diag(ml2041)
    ml2052 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2040, 1, ml2052, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2040, 2);
        view(ml2040, :, i)[:] .*= ml2051;
    end;        

    # R: ml2052, full, tmp19: ml2049, symmetric_lower_triangular, tmp32: ml2050, full, tmp29: ml2040, full
    for i = 1:2000-1;
        view(ml2049, i, i+1:2000)[:] = view(ml2049, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2052, ml2040, 1.0, ml2049)

    # tmp32: ml2050, full, tmp31: ml2049, full
    ml2053 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2049, ml2053, info) = LinearAlgebra.LAPACK.getrf!(ml2049)

    # tmp32: ml2050, full, P35: ml2053, ipiv, L33: ml2049, lower_triangular_udiag, U34: ml2049, upper_triangular
    ml2054 = [1:length(ml2053);]
    @inbounds for i in 1:length(ml2053)
        ml2054[i], ml2054[ml2053[i]] = ml2054[ml2053[i]], ml2054[i];
    end;
    ml2055 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2055 = ml2050[ml2054]

    # L33: ml2049, lower_triangular_udiag, U34: ml2049, upper_triangular, tmp40: ml2055, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2049, ml2055)

    # U34: ml2049, upper_triangular, tmp41: ml2055, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2049, ml2055)

    # tmp17: ml2055, full
    # x = tmp17
    return (ml2055)
end