using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm19(ml638::Array{Float64,2}, ml639::Array{Float64,2}, ml640::Array{Float64,2}, ml641::Array{Float64,2}, ml642::Array{Float64,1})
    # cost 5.07e+10
    # R: ml638, full, L: ml639, full, A: ml640, full, B: ml641, full, y: ml642, full
    ml643 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml643, ml641)

    # R: ml638, full, L: ml639, full, A: ml640, full, y: ml642, full, tmp26: ml643, full
    ml644 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml640, ml644, info) = LinearAlgebra.LAPACK.getrf!(ml640)

    # R: ml638, full, L: ml639, full, y: ml642, full, tmp26: ml643, full, P11: ml644, ipiv, L9: ml640, lower_triangular_udiag, U10: ml640, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml640, ml643)

    # R: ml638, full, L: ml639, full, y: ml642, full, P11: ml644, ipiv, L9: ml640, lower_triangular_udiag, tmp27: ml643, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml640, ml643)

    # R: ml638, full, L: ml639, full, y: ml642, full, P11: ml644, ipiv, tmp28: ml643, full
    ml645 = [1:length(ml644);]
    @inbounds for i in 1:length(ml644)
        ml645[i], ml645[ml644[i]] = ml645[ml644[i]], ml645[i];
    end;
    ml646 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml646 = ml643[invperm(ml645),:]

    # R: ml638, full, L: ml639, full, y: ml642, full, tmp25: ml646, full
    ml647 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml646, 0.0, ml647)

    # R: ml638, full, L: ml639, full, y: ml642, full, tmp19: ml647, symmetric_lower_triangular
    ml648 = diag(ml639)
    ml649 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml638, 1, ml649, 1)
    # tmp29 = (L R)
    for i = 1:size(ml638, 2);
        view(ml638, :, i)[:] .*= ml648;
    end;        

    # R: ml649, full, y: ml642, full, tmp19: ml647, symmetric_lower_triangular, tmp29: ml638, full
    for i = 1:2000-1;
        view(ml647, i, i+1:2000)[:] = view(ml647, i+1:2000, i);
    end;
    ml650 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml647, 1, ml650, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml638, ml649, 1.0, ml647)

    # y: ml642, full, tmp19: ml650, full, tmp31: ml647, full
    ml651 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml647, ml651, info) = LinearAlgebra.LAPACK.getrf!(ml647)

    # y: ml642, full, tmp19: ml650, full, P35: ml651, ipiv, L33: ml647, lower_triangular_udiag, U34: ml647, upper_triangular
    ml652 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml650, ml642, 0.0, ml652)

    # P35: ml651, ipiv, L33: ml647, lower_triangular_udiag, U34: ml647, upper_triangular, tmp32: ml652, full
    ml653 = [1:length(ml651);]
    @inbounds for i in 1:length(ml651)
        ml653[i], ml653[ml651[i]] = ml653[ml651[i]], ml653[i];
    end;
    ml654 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml654 = ml652[ml653]

    # L33: ml647, lower_triangular_udiag, U34: ml647, upper_triangular, tmp40: ml654, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml647, ml654)

    # U34: ml647, upper_triangular, tmp41: ml654, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml647, ml654)

    # tmp17: ml654, full
    # x = tmp17
    return (ml654)
end