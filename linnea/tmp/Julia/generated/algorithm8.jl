using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm8(ml264::Array{Float64,2}, ml265::Array{Float64,2}, ml266::Array{Float64,2}, ml267::Array{Float64,2}, ml268::Array{Float64,1})
    # cost 5.07e+10
    # R: ml264, full, L: ml265, full, A: ml266, full, B: ml267, full, y: ml268, full
    ml269 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml266, ml269, info) = LinearAlgebra.LAPACK.getrf!(ml266)

    # R: ml264, full, L: ml265, full, B: ml267, full, y: ml268, full, P11: ml269, ipiv, L9: ml266, lower_triangular_udiag, U10: ml266, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml266, ml267)

    # R: ml264, full, L: ml265, full, y: ml268, full, P11: ml269, ipiv, L9: ml266, lower_triangular_udiag, tmp53: ml267, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml266, ml267)

    # R: ml264, full, L: ml265, full, y: ml268, full, P11: ml269, ipiv, tmp54: ml267, full
    ml270 = [1:length(ml269);]
    @inbounds for i in 1:length(ml269)
        ml270[i], ml270[ml269[i]] = ml270[ml269[i]], ml270[i];
    end;
    ml271 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml271 = ml267[:,invperm(ml270)]

    # R: ml264, full, L: ml265, full, y: ml268, full, tmp55: ml271, full
    ml272 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml272, ml271)

    # R: ml264, full, L: ml265, full, y: ml268, full, tmp25: ml272, full
    ml273 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml272, 0.0, ml273)

    # R: ml264, full, L: ml265, full, y: ml268, full, tmp19: ml273, symmetric_lower_triangular
    ml274 = diag(ml265)
    ml275 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml264, 1, ml275, 1)
    # tmp29 = (L R)
    for i = 1:size(ml264, 2);
        view(ml264, :, i)[:] .*= ml274;
    end;        

    # R: ml275, full, y: ml268, full, tmp19: ml273, symmetric_lower_triangular, tmp29: ml264, full
    for i = 1:2000-1;
        view(ml273, i, i+1:2000)[:] = view(ml273, i+1:2000, i);
    end;
    ml276 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml273, 1, ml276, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml264, ml275, 1.0, ml273)

    # y: ml268, full, tmp19: ml276, full, tmp31: ml273, full
    ml277 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml273, ml277, info) = LinearAlgebra.LAPACK.getrf!(ml273)

    # y: ml268, full, tmp19: ml276, full, P35: ml277, ipiv, L33: ml273, lower_triangular_udiag, U34: ml273, upper_triangular
    ml278 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml276, ml268, 0.0, ml278)

    # P35: ml277, ipiv, L33: ml273, lower_triangular_udiag, U34: ml273, upper_triangular, tmp32: ml278, full
    ml279 = [1:length(ml277);]
    @inbounds for i in 1:length(ml277)
        ml279[i], ml279[ml277[i]] = ml279[ml277[i]], ml279[i];
    end;
    ml280 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml280 = ml278[ml279]

    # L33: ml273, lower_triangular_udiag, U34: ml273, upper_triangular, tmp40: ml280, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml273, ml280)

    # U34: ml273, upper_triangular, tmp41: ml280, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml273, ml280)

    # tmp17: ml280, full
    # x = tmp17
    return (ml280)
end