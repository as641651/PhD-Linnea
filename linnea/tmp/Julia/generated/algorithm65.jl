using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm65(ml2170::Array{Float64,2}, ml2171::Array{Float64,2}, ml2172::Array{Float64,2}, ml2173::Array{Float64,2}, ml2174::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2170, full, L: ml2171, full, A: ml2172, full, B: ml2173, full, y: ml2174, full
    ml2175 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2175, ml2173)

    # R: ml2170, full, L: ml2171, full, A: ml2172, full, y: ml2174, full, tmp26: ml2175, full
    ml2176 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2172, ml2176, info) = LinearAlgebra.LAPACK.getrf!(ml2172)

    # R: ml2170, full, L: ml2171, full, y: ml2174, full, tmp26: ml2175, full, P11: ml2176, ipiv, L9: ml2172, lower_triangular_udiag, U10: ml2172, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2172, ml2175)

    # R: ml2170, full, L: ml2171, full, y: ml2174, full, P11: ml2176, ipiv, L9: ml2172, lower_triangular_udiag, tmp27: ml2175, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2172, ml2175)

    # R: ml2170, full, L: ml2171, full, y: ml2174, full, P11: ml2176, ipiv, tmp28: ml2175, full
    ml2177 = [1:length(ml2176);]
    @inbounds for i in 1:length(ml2176)
        ml2177[i], ml2177[ml2176[i]] = ml2177[ml2176[i]], ml2177[i];
    end;
    ml2178 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2178 = ml2175[invperm(ml2177),:]

    # R: ml2170, full, L: ml2171, full, y: ml2174, full, tmp25: ml2178, full
    ml2179 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2178, 0.0, ml2179)

    # R: ml2170, full, L: ml2171, full, y: ml2174, full, tmp19: ml2179, symmetric_lower_triangular
    ml2180 = diag(ml2171)
    ml2181 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2170, 1, ml2181, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2170, 2);
        view(ml2170, :, i)[:] .*= ml2180;
    end;        

    # R: ml2181, full, y: ml2174, full, tmp19: ml2179, symmetric_lower_triangular, tmp29: ml2170, full
    for i = 1:2000-1;
        view(ml2179, i, i+1:2000)[:] = view(ml2179, i+1:2000, i);
    end;
    ml2182 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2179, 1, ml2182, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2170, ml2181, 1.0, ml2179)

    # y: ml2174, full, tmp19: ml2182, full, tmp31: ml2179, full
    ml2183 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2179, ml2183, info) = LinearAlgebra.LAPACK.getrf!(ml2179)

    # y: ml2174, full, tmp19: ml2182, full, P35: ml2183, ipiv, L33: ml2179, lower_triangular_udiag, U34: ml2179, upper_triangular
    ml2184 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2182, ml2174, 0.0, ml2184)

    # P35: ml2183, ipiv, L33: ml2179, lower_triangular_udiag, U34: ml2179, upper_triangular, tmp32: ml2184, full
    ml2185 = [1:length(ml2183);]
    @inbounds for i in 1:length(ml2183)
        ml2185[i], ml2185[ml2183[i]] = ml2185[ml2183[i]], ml2185[i];
    end;
    ml2186 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2186 = ml2184[ml2185]

    # L33: ml2179, lower_triangular_udiag, U34: ml2179, upper_triangular, tmp40: ml2186, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2179, ml2186)

    # U34: ml2179, upper_triangular, tmp41: ml2186, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2179, ml2186)

    # tmp17: ml2186, full
    # x = tmp17
    return (ml2186)
end