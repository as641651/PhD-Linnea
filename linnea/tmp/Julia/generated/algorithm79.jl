using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm79(ml2646::Array{Float64,2}, ml2647::Array{Float64,2}, ml2648::Array{Float64,2}, ml2649::Array{Float64,2}, ml2650::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2646, full, L: ml2647, full, A: ml2648, full, B: ml2649, full, y: ml2650, full
    ml2651 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2651, ml2649)

    # R: ml2646, full, L: ml2647, full, A: ml2648, full, y: ml2650, full, tmp26: ml2651, full
    ml2652 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2648, ml2652, info) = LinearAlgebra.LAPACK.getrf!(ml2648)

    # R: ml2646, full, L: ml2647, full, y: ml2650, full, tmp26: ml2651, full, P11: ml2652, ipiv, L9: ml2648, lower_triangular_udiag, U10: ml2648, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2648, ml2651)

    # R: ml2646, full, L: ml2647, full, y: ml2650, full, P11: ml2652, ipiv, L9: ml2648, lower_triangular_udiag, tmp27: ml2651, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2648, ml2651)

    # R: ml2646, full, L: ml2647, full, y: ml2650, full, P11: ml2652, ipiv, tmp28: ml2651, full
    ml2653 = [1:length(ml2652);]
    @inbounds for i in 1:length(ml2652)
        ml2653[i], ml2653[ml2652[i]] = ml2653[ml2652[i]], ml2653[i];
    end;
    ml2654 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2654 = ml2651[invperm(ml2653),:]

    # R: ml2646, full, L: ml2647, full, y: ml2650, full, tmp25: ml2654, full
    ml2655 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2654, 0.0, ml2655)

    # R: ml2646, full, L: ml2647, full, y: ml2650, full, tmp19: ml2655, symmetric_lower_triangular
    ml2656 = diag(ml2647)
    ml2657 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2646, 1, ml2657, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2646, 2);
        view(ml2646, :, i)[:] .*= ml2656;
    end;        

    # R: ml2657, full, y: ml2650, full, tmp19: ml2655, symmetric_lower_triangular, tmp29: ml2646, full
    for i = 1:2000-1;
        view(ml2655, i, i+1:2000)[:] = view(ml2655, i+1:2000, i);
    end;
    ml2658 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2655, 1, ml2658, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2646, ml2657, 1.0, ml2655)

    # y: ml2650, full, tmp19: ml2658, full, tmp31: ml2655, full
    ml2659 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2655, ml2659, info) = LinearAlgebra.LAPACK.getrf!(ml2655)

    # y: ml2650, full, tmp19: ml2658, full, P35: ml2659, ipiv, L33: ml2655, lower_triangular_udiag, U34: ml2655, upper_triangular
    ml2660 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2658, ml2650, 0.0, ml2660)

    # P35: ml2659, ipiv, L33: ml2655, lower_triangular_udiag, U34: ml2655, upper_triangular, tmp32: ml2660, full
    ml2661 = [1:length(ml2659);]
    @inbounds for i in 1:length(ml2659)
        ml2661[i], ml2661[ml2659[i]] = ml2661[ml2659[i]], ml2661[i];
    end;
    ml2662 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2662 = ml2660[ml2661]

    # L33: ml2655, lower_triangular_udiag, U34: ml2655, upper_triangular, tmp40: ml2662, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2655, ml2662)

    # U34: ml2655, upper_triangular, tmp41: ml2662, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2655, ml2662)

    # tmp17: ml2662, full
    # x = tmp17
    return (ml2662)
end