using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm62(ml2072::Array{Float64,2}, ml2073::Array{Float64,2}, ml2074::Array{Float64,2}, ml2075::Array{Float64,2}, ml2076::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2072, full, L: ml2073, full, A: ml2074, full, B: ml2075, full, y: ml2076, full
    ml2077 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2077, ml2075)

    # R: ml2072, full, L: ml2073, full, A: ml2074, full, y: ml2076, full, tmp26: ml2077, full
    ml2078 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2074, ml2078, info) = LinearAlgebra.LAPACK.getrf!(ml2074)

    # R: ml2072, full, L: ml2073, full, y: ml2076, full, tmp26: ml2077, full, P11: ml2078, ipiv, L9: ml2074, lower_triangular_udiag, U10: ml2074, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2074, ml2077)

    # R: ml2072, full, L: ml2073, full, y: ml2076, full, P11: ml2078, ipiv, L9: ml2074, lower_triangular_udiag, tmp27: ml2077, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2074, ml2077)

    # R: ml2072, full, L: ml2073, full, y: ml2076, full, P11: ml2078, ipiv, tmp28: ml2077, full
    ml2079 = [1:length(ml2078);]
    @inbounds for i in 1:length(ml2078)
        ml2079[i], ml2079[ml2078[i]] = ml2079[ml2078[i]], ml2079[i];
    end;
    ml2080 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2080 = ml2077[invperm(ml2079),:]

    # R: ml2072, full, L: ml2073, full, y: ml2076, full, tmp25: ml2080, full
    ml2081 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2080, 0.0, ml2081)

    # R: ml2072, full, L: ml2073, full, y: ml2076, full, tmp19: ml2081, symmetric_lower_triangular
    ml2082 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2081, ml2076, 0.0, ml2082)

    # R: ml2072, full, L: ml2073, full, tmp19: ml2081, symmetric_lower_triangular, tmp32: ml2082, full
    ml2083 = diag(ml2073)
    ml2084 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2072, 1, ml2084, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2072, 2);
        view(ml2072, :, i)[:] .*= ml2083;
    end;        

    # R: ml2084, full, tmp19: ml2081, symmetric_lower_triangular, tmp32: ml2082, full, tmp29: ml2072, full
    for i = 1:2000-1;
        view(ml2081, i, i+1:2000)[:] = view(ml2081, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2084, ml2072, 1.0, ml2081)

    # tmp32: ml2082, full, tmp31: ml2081, full
    ml2085 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2081, ml2085, info) = LinearAlgebra.LAPACK.getrf!(ml2081)

    # tmp32: ml2082, full, P35: ml2085, ipiv, L33: ml2081, lower_triangular_udiag, U34: ml2081, upper_triangular
    ml2086 = [1:length(ml2085);]
    @inbounds for i in 1:length(ml2085)
        ml2086[i], ml2086[ml2085[i]] = ml2086[ml2085[i]], ml2086[i];
    end;
    ml2087 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2087 = ml2082[ml2086]

    # L33: ml2081, lower_triangular_udiag, U34: ml2081, upper_triangular, tmp40: ml2087, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2081, ml2087)

    # U34: ml2081, upper_triangular, tmp41: ml2087, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2081, ml2087)

    # tmp17: ml2087, full
    # x = tmp17
    return (ml2087)
end