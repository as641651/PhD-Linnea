using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm64(ml2136::Array{Float64,2}, ml2137::Array{Float64,2}, ml2138::Array{Float64,2}, ml2139::Array{Float64,2}, ml2140::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2136, full, L: ml2137, full, A: ml2138, full, B: ml2139, full, y: ml2140, full
    ml2141 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2141, ml2139)

    # R: ml2136, full, L: ml2137, full, A: ml2138, full, y: ml2140, full, tmp26: ml2141, full
    ml2142 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2138, ml2142, info) = LinearAlgebra.LAPACK.getrf!(ml2138)

    # R: ml2136, full, L: ml2137, full, y: ml2140, full, tmp26: ml2141, full, P11: ml2142, ipiv, L9: ml2138, lower_triangular_udiag, U10: ml2138, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2138, ml2141)

    # R: ml2136, full, L: ml2137, full, y: ml2140, full, P11: ml2142, ipiv, L9: ml2138, lower_triangular_udiag, tmp27: ml2141, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2138, ml2141)

    # R: ml2136, full, L: ml2137, full, y: ml2140, full, P11: ml2142, ipiv, tmp28: ml2141, full
    ml2143 = [1:length(ml2142);]
    @inbounds for i in 1:length(ml2142)
        ml2143[i], ml2143[ml2142[i]] = ml2143[ml2142[i]], ml2143[i];
    end;
    ml2144 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2144 = ml2141[invperm(ml2143),:]

    # R: ml2136, full, L: ml2137, full, y: ml2140, full, tmp25: ml2144, full
    ml2145 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2144, 0.0, ml2145)

    # R: ml2136, full, L: ml2137, full, y: ml2140, full, tmp19: ml2145, symmetric_lower_triangular
    ml2146 = diag(ml2137)
    ml2147 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2136, 1, ml2147, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2136, 2);
        view(ml2136, :, i)[:] .*= ml2146;
    end;        

    # R: ml2147, full, y: ml2140, full, tmp19: ml2145, symmetric_lower_triangular, tmp29: ml2136, full
    for i = 1:2000-1;
        view(ml2145, i, i+1:2000)[:] = view(ml2145, i+1:2000, i);
    end;
    ml2148 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2145, 1, ml2148, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2136, ml2147, 1.0, ml2145)

    # y: ml2140, full, tmp19: ml2148, full, tmp31: ml2145, full
    ml2149 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2145, ml2149, info) = LinearAlgebra.LAPACK.getrf!(ml2145)

    # y: ml2140, full, tmp19: ml2148, full, P35: ml2149, ipiv, L33: ml2145, lower_triangular_udiag, U34: ml2145, upper_triangular
    ml2150 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2148, ml2140, 0.0, ml2150)

    # P35: ml2149, ipiv, L33: ml2145, lower_triangular_udiag, U34: ml2145, upper_triangular, tmp32: ml2150, full
    ml2151 = [1:length(ml2149);]
    @inbounds for i in 1:length(ml2149)
        ml2151[i], ml2151[ml2149[i]] = ml2151[ml2149[i]], ml2151[i];
    end;
    ml2152 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2152 = ml2150[ml2151]

    # L33: ml2145, lower_triangular_udiag, U34: ml2145, upper_triangular, tmp40: ml2152, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2145, ml2152)

    # U34: ml2145, upper_triangular, tmp41: ml2152, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2145, ml2152)

    # tmp17: ml2152, full
    # x = tmp17
    return (ml2152)
end