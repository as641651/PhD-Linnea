using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm66(ml2204::Array{Float64,2}, ml2205::Array{Float64,2}, ml2206::Array{Float64,2}, ml2207::Array{Float64,2}, ml2208::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2204, full, L: ml2205, full, A: ml2206, full, B: ml2207, full, y: ml2208, full
    ml2209 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2209, ml2207)

    # R: ml2204, full, L: ml2205, full, A: ml2206, full, y: ml2208, full, tmp26: ml2209, full
    ml2210 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2206, ml2210, info) = LinearAlgebra.LAPACK.getrf!(ml2206)

    # R: ml2204, full, L: ml2205, full, y: ml2208, full, tmp26: ml2209, full, P11: ml2210, ipiv, L9: ml2206, lower_triangular_udiag, U10: ml2206, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2206, ml2209)

    # R: ml2204, full, L: ml2205, full, y: ml2208, full, P11: ml2210, ipiv, L9: ml2206, lower_triangular_udiag, tmp27: ml2209, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2206, ml2209)

    # R: ml2204, full, L: ml2205, full, y: ml2208, full, P11: ml2210, ipiv, tmp28: ml2209, full
    ml2211 = [1:length(ml2210);]
    @inbounds for i in 1:length(ml2210)
        ml2211[i], ml2211[ml2210[i]] = ml2211[ml2210[i]], ml2211[i];
    end;
    ml2212 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2212 = ml2209[invperm(ml2211),:]

    # R: ml2204, full, L: ml2205, full, y: ml2208, full, tmp25: ml2212, full
    ml2213 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2212, 0.0, ml2213)

    # R: ml2204, full, L: ml2205, full, y: ml2208, full, tmp19: ml2213, symmetric_lower_triangular
    ml2214 = diag(ml2205)
    ml2215 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2204, 1, ml2215, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2204, 2);
        view(ml2204, :, i)[:] .*= ml2214;
    end;        

    # R: ml2215, full, y: ml2208, full, tmp19: ml2213, symmetric_lower_triangular, tmp29: ml2204, full
    for i = 1:2000-1;
        view(ml2213, i, i+1:2000)[:] = view(ml2213, i+1:2000, i);
    end;
    ml2216 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2213, 1, ml2216, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2204, ml2215, 1.0, ml2213)

    # y: ml2208, full, tmp19: ml2216, full, tmp31: ml2213, full
    ml2217 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2213, ml2217, info) = LinearAlgebra.LAPACK.getrf!(ml2213)

    # y: ml2208, full, tmp19: ml2216, full, P35: ml2217, ipiv, L33: ml2213, lower_triangular_udiag, U34: ml2213, upper_triangular
    ml2218 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2216, ml2208, 0.0, ml2218)

    # P35: ml2217, ipiv, L33: ml2213, lower_triangular_udiag, U34: ml2213, upper_triangular, tmp32: ml2218, full
    ml2219 = [1:length(ml2217);]
    @inbounds for i in 1:length(ml2217)
        ml2219[i], ml2219[ml2217[i]] = ml2219[ml2217[i]], ml2219[i];
    end;
    ml2220 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2220 = ml2218[ml2219]

    # L33: ml2213, lower_triangular_udiag, U34: ml2213, upper_triangular, tmp40: ml2220, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2213, ml2220)

    # U34: ml2213, upper_triangular, tmp41: ml2220, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2213, ml2220)

    # tmp17: ml2220, full
    # x = tmp17
    return (ml2220)
end