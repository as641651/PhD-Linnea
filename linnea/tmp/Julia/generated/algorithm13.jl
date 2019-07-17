using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm13(ml434::Array{Float64,2}, ml435::Array{Float64,2}, ml436::Array{Float64,2}, ml437::Array{Float64,2}, ml438::Array{Float64,1})
    # cost 5.07e+10
    # R: ml434, full, L: ml435, full, A: ml436, full, B: ml437, full, y: ml438, full
    ml439 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml436, ml439, info) = LinearAlgebra.LAPACK.getrf!(ml436)

    # R: ml434, full, L: ml435, full, B: ml437, full, y: ml438, full, P11: ml439, ipiv, L9: ml436, lower_triangular_udiag, U10: ml436, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml436, ml437)

    # R: ml434, full, L: ml435, full, y: ml438, full, P11: ml439, ipiv, L9: ml436, lower_triangular_udiag, tmp53: ml437, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml436, ml437)

    # R: ml434, full, L: ml435, full, y: ml438, full, P11: ml439, ipiv, tmp54: ml437, full
    ml440 = [1:length(ml439);]
    @inbounds for i in 1:length(ml439)
        ml440[i], ml440[ml439[i]] = ml440[ml439[i]], ml440[i];
    end;
    ml441 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml441 = ml437[:,invperm(ml440)]

    # R: ml434, full, L: ml435, full, y: ml438, full, tmp55: ml441, full
    ml442 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml442, ml441)

    # R: ml434, full, L: ml435, full, y: ml438, full, tmp25: ml442, full
    ml443 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml442, 0.0, ml443)

    # R: ml434, full, L: ml435, full, y: ml438, full, tmp19: ml443, symmetric_lower_triangular
    ml444 = diag(ml435)
    ml445 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml434, 1, ml445, 1)
    # tmp29 = (L R)
    for i = 1:size(ml434, 2);
        view(ml434, :, i)[:] .*= ml444;
    end;        

    # R: ml445, full, y: ml438, full, tmp19: ml443, symmetric_lower_triangular, tmp29: ml434, full
    for i = 1:2000-1;
        view(ml443, i, i+1:2000)[:] = view(ml443, i+1:2000, i);
    end;
    ml446 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml443, 1, ml446, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml434, ml445, 1.0, ml443)

    # y: ml438, full, tmp19: ml446, full, tmp31: ml443, full
    ml447 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml446, ml438, 0.0, ml447)

    # tmp31: ml443, full, tmp32: ml447, full
    ml448 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml443, ml448, info) = LinearAlgebra.LAPACK.getrf!(ml443)

    # tmp32: ml447, full, P35: ml448, ipiv, L33: ml443, lower_triangular_udiag, U34: ml443, upper_triangular
    ml449 = [1:length(ml448);]
    @inbounds for i in 1:length(ml448)
        ml449[i], ml449[ml448[i]] = ml449[ml448[i]], ml449[i];
    end;
    ml450 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml450 = ml447[ml449]

    # L33: ml443, lower_triangular_udiag, U34: ml443, upper_triangular, tmp40: ml450, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml443, ml450)

    # U34: ml443, upper_triangular, tmp41: ml450, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml443, ml450)

    # tmp17: ml450, full
    # x = tmp17
    return (ml450)
end