using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm11(ml366::Array{Float64,2}, ml367::Array{Float64,2}, ml368::Array{Float64,2}, ml369::Array{Float64,2}, ml370::Array{Float64,1})
    # cost 5.07e+10
    # R: ml366, full, L: ml367, full, A: ml368, full, B: ml369, full, y: ml370, full
    ml371 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml368, ml371, info) = LinearAlgebra.LAPACK.getrf!(ml368)

    # R: ml366, full, L: ml367, full, B: ml369, full, y: ml370, full, P11: ml371, ipiv, L9: ml368, lower_triangular_udiag, U10: ml368, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml368, ml369)

    # R: ml366, full, L: ml367, full, y: ml370, full, P11: ml371, ipiv, L9: ml368, lower_triangular_udiag, tmp53: ml369, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml368, ml369)

    # R: ml366, full, L: ml367, full, y: ml370, full, P11: ml371, ipiv, tmp54: ml369, full
    ml372 = [1:length(ml371);]
    @inbounds for i in 1:length(ml371)
        ml372[i], ml372[ml371[i]] = ml372[ml371[i]], ml372[i];
    end;
    ml373 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml373 = ml369[:,invperm(ml372)]

    # R: ml366, full, L: ml367, full, y: ml370, full, tmp55: ml373, full
    ml374 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml374, ml373)

    # R: ml366, full, L: ml367, full, y: ml370, full, tmp25: ml374, full
    ml375 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml374, 0.0, ml375)

    # R: ml366, full, L: ml367, full, y: ml370, full, tmp19: ml375, symmetric_lower_triangular
    ml376 = diag(ml367)
    ml377 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml366, 1, ml377, 1)
    # tmp29 = (L R)
    for i = 1:size(ml366, 2);
        view(ml366, :, i)[:] .*= ml376;
    end;        

    # R: ml377, full, y: ml370, full, tmp19: ml375, symmetric_lower_triangular, tmp29: ml366, full
    for i = 1:2000-1;
        view(ml375, i, i+1:2000)[:] = view(ml375, i+1:2000, i);
    end;
    ml378 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml375, 1, ml378, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml366, ml377, 1.0, ml375)

    # y: ml370, full, tmp19: ml378, full, tmp31: ml375, full
    ml379 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml378, ml370, 0.0, ml379)

    # tmp31: ml375, full, tmp32: ml379, full
    ml380 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml375, ml380, info) = LinearAlgebra.LAPACK.getrf!(ml375)

    # tmp32: ml379, full, P35: ml380, ipiv, L33: ml375, lower_triangular_udiag, U34: ml375, upper_triangular
    ml381 = [1:length(ml380);]
    @inbounds for i in 1:length(ml380)
        ml381[i], ml381[ml380[i]] = ml381[ml380[i]], ml381[i];
    end;
    ml382 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml382 = ml379[ml381]

    # L33: ml375, lower_triangular_udiag, U34: ml375, upper_triangular, tmp40: ml382, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml375, ml382)

    # U34: ml375, upper_triangular, tmp41: ml382, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml375, ml382)

    # tmp17: ml382, full
    # x = tmp17
    return (ml382)
end