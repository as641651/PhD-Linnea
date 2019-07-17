using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm80(ml2680::Array{Float64,2}, ml2681::Array{Float64,2}, ml2682::Array{Float64,2}, ml2683::Array{Float64,2}, ml2684::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2680, full, L: ml2681, full, A: ml2682, full, B: ml2683, full, y: ml2684, full
    ml2685 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2685, ml2683)

    # R: ml2680, full, L: ml2681, full, A: ml2682, full, y: ml2684, full, tmp26: ml2685, full
    ml2686 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2682, ml2686, info) = LinearAlgebra.LAPACK.getrf!(ml2682)

    # R: ml2680, full, L: ml2681, full, y: ml2684, full, tmp26: ml2685, full, P11: ml2686, ipiv, L9: ml2682, lower_triangular_udiag, U10: ml2682, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2682, ml2685)

    # R: ml2680, full, L: ml2681, full, y: ml2684, full, P11: ml2686, ipiv, L9: ml2682, lower_triangular_udiag, tmp27: ml2685, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2682, ml2685)

    # R: ml2680, full, L: ml2681, full, y: ml2684, full, P11: ml2686, ipiv, tmp28: ml2685, full
    ml2687 = [1:length(ml2686);]
    @inbounds for i in 1:length(ml2686)
        ml2687[i], ml2687[ml2686[i]] = ml2687[ml2686[i]], ml2687[i];
    end;
    ml2688 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2688 = ml2685[invperm(ml2687),:]

    # R: ml2680, full, L: ml2681, full, y: ml2684, full, tmp25: ml2688, full
    ml2689 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2688, 0.0, ml2689)

    # R: ml2680, full, L: ml2681, full, y: ml2684, full, tmp19: ml2689, symmetric_lower_triangular
    ml2690 = diag(ml2681)
    ml2691 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2680, 1, ml2691, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2680, 2);
        view(ml2680, :, i)[:] .*= ml2690;
    end;        

    # R: ml2691, full, y: ml2684, full, tmp19: ml2689, symmetric_lower_triangular, tmp29: ml2680, full
    ml2692 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2689, ml2684, 0.0, ml2692)

    # R: ml2691, full, tmp19: ml2689, symmetric_lower_triangular, tmp29: ml2680, full, tmp32: ml2692, full
    for i = 1:2000-1;
        view(ml2689, i, i+1:2000)[:] = view(ml2689, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2680, ml2691, 1.0, ml2689)

    # tmp32: ml2692, full, tmp31: ml2689, full
    ml2693 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2689, ml2693, info) = LinearAlgebra.LAPACK.getrf!(ml2689)

    # tmp32: ml2692, full, P35: ml2693, ipiv, L33: ml2689, lower_triangular_udiag, U34: ml2689, upper_triangular
    ml2694 = [1:length(ml2693);]
    @inbounds for i in 1:length(ml2693)
        ml2694[i], ml2694[ml2693[i]] = ml2694[ml2693[i]], ml2694[i];
    end;
    ml2695 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2695 = ml2692[ml2694]

    # L33: ml2689, lower_triangular_udiag, U34: ml2689, upper_triangular, tmp40: ml2695, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2689, ml2695)

    # U34: ml2689, upper_triangular, tmp41: ml2695, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2689, ml2695)

    # tmp17: ml2695, full
    # x = tmp17
    return (ml2695)
end