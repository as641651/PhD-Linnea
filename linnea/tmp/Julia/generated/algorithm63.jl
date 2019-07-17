using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm63(ml2104::Array{Float64,2}, ml2105::Array{Float64,2}, ml2106::Array{Float64,2}, ml2107::Array{Float64,2}, ml2108::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2104, full, L: ml2105, full, A: ml2106, full, B: ml2107, full, y: ml2108, full
    ml2109 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2109, ml2107)

    # R: ml2104, full, L: ml2105, full, A: ml2106, full, y: ml2108, full, tmp26: ml2109, full
    ml2110 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2106, ml2110, info) = LinearAlgebra.LAPACK.getrf!(ml2106)

    # R: ml2104, full, L: ml2105, full, y: ml2108, full, tmp26: ml2109, full, P11: ml2110, ipiv, L9: ml2106, lower_triangular_udiag, U10: ml2106, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2106, ml2109)

    # R: ml2104, full, L: ml2105, full, y: ml2108, full, P11: ml2110, ipiv, L9: ml2106, lower_triangular_udiag, tmp27: ml2109, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2106, ml2109)

    # R: ml2104, full, L: ml2105, full, y: ml2108, full, P11: ml2110, ipiv, tmp28: ml2109, full
    ml2111 = [1:length(ml2110);]
    @inbounds for i in 1:length(ml2110)
        ml2111[i], ml2111[ml2110[i]] = ml2111[ml2110[i]], ml2111[i];
    end;
    ml2112 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2112 = ml2109[invperm(ml2111),:]

    # R: ml2104, full, L: ml2105, full, y: ml2108, full, tmp25: ml2112, full
    ml2113 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2112, 0.0, ml2113)

    # R: ml2104, full, L: ml2105, full, y: ml2108, full, tmp19: ml2113, symmetric_lower_triangular
    ml2114 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2113, ml2108, 0.0, ml2114)

    # R: ml2104, full, L: ml2105, full, tmp19: ml2113, symmetric_lower_triangular, tmp32: ml2114, full
    ml2115 = diag(ml2105)
    ml2116 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2104, 1, ml2116, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2104, 2);
        view(ml2104, :, i)[:] .*= ml2115;
    end;        

    # R: ml2116, full, tmp19: ml2113, symmetric_lower_triangular, tmp32: ml2114, full, tmp29: ml2104, full
    for i = 1:2000-1;
        view(ml2113, i, i+1:2000)[:] = view(ml2113, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2116, ml2104, 1.0, ml2113)

    # tmp32: ml2114, full, tmp31: ml2113, full
    ml2117 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2113, ml2117, info) = LinearAlgebra.LAPACK.getrf!(ml2113)

    # tmp32: ml2114, full, P35: ml2117, ipiv, L33: ml2113, lower_triangular_udiag, U34: ml2113, upper_triangular
    ml2118 = [1:length(ml2117);]
    @inbounds for i in 1:length(ml2117)
        ml2118[i], ml2118[ml2117[i]] = ml2118[ml2117[i]], ml2118[i];
    end;
    ml2119 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2119 = ml2114[ml2118]

    # L33: ml2113, lower_triangular_udiag, U34: ml2113, upper_triangular, tmp40: ml2119, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2113, ml2119)

    # U34: ml2113, upper_triangular, tmp41: ml2119, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2113, ml2119)

    # tmp17: ml2119, full
    # x = tmp17
    return (ml2119)
end