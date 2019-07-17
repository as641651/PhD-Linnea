using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm81(ml2712::Array{Float64,2}, ml2713::Array{Float64,2}, ml2714::Array{Float64,2}, ml2715::Array{Float64,2}, ml2716::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2712, full, L: ml2713, full, A: ml2714, full, B: ml2715, full, y: ml2716, full
    ml2717 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2717, ml2715)

    # R: ml2712, full, L: ml2713, full, A: ml2714, full, y: ml2716, full, tmp26: ml2717, full
    ml2718 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2714, ml2718, info) = LinearAlgebra.LAPACK.getrf!(ml2714)

    # R: ml2712, full, L: ml2713, full, y: ml2716, full, tmp26: ml2717, full, P11: ml2718, ipiv, L9: ml2714, lower_triangular_udiag, U10: ml2714, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2714, ml2717)

    # R: ml2712, full, L: ml2713, full, y: ml2716, full, P11: ml2718, ipiv, L9: ml2714, lower_triangular_udiag, tmp27: ml2717, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2714, ml2717)

    # R: ml2712, full, L: ml2713, full, y: ml2716, full, P11: ml2718, ipiv, tmp28: ml2717, full
    ml2719 = [1:length(ml2718);]
    @inbounds for i in 1:length(ml2718)
        ml2719[i], ml2719[ml2718[i]] = ml2719[ml2718[i]], ml2719[i];
    end;
    ml2720 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2720 = ml2717[invperm(ml2719),:]

    # R: ml2712, full, L: ml2713, full, y: ml2716, full, tmp25: ml2720, full
    ml2721 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2720, 0.0, ml2721)

    # R: ml2712, full, L: ml2713, full, y: ml2716, full, tmp19: ml2721, symmetric_lower_triangular
    ml2722 = diag(ml2713)
    ml2723 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2712, 1, ml2723, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2712, 2);
        view(ml2712, :, i)[:] .*= ml2722;
    end;        

    # R: ml2723, full, y: ml2716, full, tmp19: ml2721, symmetric_lower_triangular, tmp29: ml2712, full
    ml2724 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2721, ml2716, 0.0, ml2724)

    # R: ml2723, full, tmp19: ml2721, symmetric_lower_triangular, tmp29: ml2712, full, tmp32: ml2724, full
    for i = 1:2000-1;
        view(ml2721, i, i+1:2000)[:] = view(ml2721, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2712, ml2723, 1.0, ml2721)

    # tmp32: ml2724, full, tmp31: ml2721, full
    ml2725 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2721, ml2725, info) = LinearAlgebra.LAPACK.getrf!(ml2721)

    # tmp32: ml2724, full, P35: ml2725, ipiv, L33: ml2721, lower_triangular_udiag, U34: ml2721, upper_triangular
    ml2726 = [1:length(ml2725);]
    @inbounds for i in 1:length(ml2725)
        ml2726[i], ml2726[ml2725[i]] = ml2726[ml2725[i]], ml2726[i];
    end;
    ml2727 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2727 = ml2724[ml2726]

    # L33: ml2721, lower_triangular_udiag, U34: ml2721, upper_triangular, tmp40: ml2727, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2721, ml2727)

    # U34: ml2721, upper_triangular, tmp41: ml2727, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2721, ml2727)

    # tmp17: ml2727, full
    # x = tmp17
    return (ml2727)
end