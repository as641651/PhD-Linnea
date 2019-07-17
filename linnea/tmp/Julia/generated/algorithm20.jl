using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm20(ml672::Array{Float64,2}, ml673::Array{Float64,2}, ml674::Array{Float64,2}, ml675::Array{Float64,2}, ml676::Array{Float64,1})
    # cost 5.07e+10
    # R: ml672, full, L: ml673, full, A: ml674, full, B: ml675, full, y: ml676, full
    ml677 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml674, ml677, info) = LinearAlgebra.LAPACK.getrf!(ml674)

    # R: ml672, full, L: ml673, full, B: ml675, full, y: ml676, full, P11: ml677, ipiv, L9: ml674, lower_triangular_udiag, U10: ml674, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml674, ml675)

    # R: ml672, full, L: ml673, full, y: ml676, full, P11: ml677, ipiv, L9: ml674, lower_triangular_udiag, tmp53: ml675, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml674, ml675)

    # R: ml672, full, L: ml673, full, y: ml676, full, P11: ml677, ipiv, tmp54: ml675, full
    ml678 = [1:length(ml677);]
    @inbounds for i in 1:length(ml677)
        ml678[i], ml678[ml677[i]] = ml678[ml677[i]], ml678[i];
    end;
    ml679 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml679 = ml675[:,invperm(ml678)]

    # R: ml672, full, L: ml673, full, y: ml676, full, tmp55: ml679, full
    ml680 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml680, ml679)

    # R: ml672, full, L: ml673, full, y: ml676, full, tmp25: ml680, full
    ml681 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml680, 0.0, ml681)

    # R: ml672, full, L: ml673, full, y: ml676, full, tmp19: ml681, symmetric_lower_triangular
    ml682 = diag(ml673)
    ml683 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml672, 1, ml683, 1)
    # tmp29 = (L R)
    for i = 1:size(ml672, 2);
        view(ml672, :, i)[:] .*= ml682;
    end;        

    # R: ml683, full, y: ml676, full, tmp19: ml681, symmetric_lower_triangular, tmp29: ml672, full
    ml684 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml681, ml676, 0.0, ml684)

    # R: ml683, full, tmp19: ml681, symmetric_lower_triangular, tmp29: ml672, full, tmp32: ml684, full
    for i = 1:2000-1;
        view(ml681, i, i+1:2000)[:] = view(ml681, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml672, ml683, 1.0, ml681)

    # tmp32: ml684, full, tmp31: ml681, full
    ml685 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml681, ml685, info) = LinearAlgebra.LAPACK.getrf!(ml681)

    # tmp32: ml684, full, P35: ml685, ipiv, L33: ml681, lower_triangular_udiag, U34: ml681, upper_triangular
    ml686 = [1:length(ml685);]
    @inbounds for i in 1:length(ml685)
        ml686[i], ml686[ml685[i]] = ml686[ml685[i]], ml686[i];
    end;
    ml687 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml687 = ml684[ml686]

    # L33: ml681, lower_triangular_udiag, U34: ml681, upper_triangular, tmp40: ml687, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml681, ml687)

    # U34: ml681, upper_triangular, tmp41: ml687, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml681, ml687)

    # tmp17: ml687, full
    # x = tmp17
    return (ml687)
end