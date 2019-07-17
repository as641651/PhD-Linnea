using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm25(ml834::Array{Float64,2}, ml835::Array{Float64,2}, ml836::Array{Float64,2}, ml837::Array{Float64,2}, ml838::Array{Float64,1})
    # cost 5.07e+10
    # R: ml834, full, L: ml835, full, A: ml836, full, B: ml837, full, y: ml838, full
    ml839 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml839, ml837)

    # R: ml834, full, L: ml835, full, A: ml836, full, y: ml838, full, tmp26: ml839, full
    ml840 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml836, ml840, info) = LinearAlgebra.LAPACK.getrf!(ml836)

    # R: ml834, full, L: ml835, full, y: ml838, full, tmp26: ml839, full, P11: ml840, ipiv, L9: ml836, lower_triangular_udiag, U10: ml836, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml836, ml839)

    # R: ml834, full, L: ml835, full, y: ml838, full, P11: ml840, ipiv, L9: ml836, lower_triangular_udiag, tmp27: ml839, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml836, ml839)

    # R: ml834, full, L: ml835, full, y: ml838, full, P11: ml840, ipiv, tmp28: ml839, full
    ml841 = [1:length(ml840);]
    @inbounds for i in 1:length(ml840)
        ml841[i], ml841[ml840[i]] = ml841[ml840[i]], ml841[i];
    end;
    ml842 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml842 = ml839[invperm(ml841),:]

    # R: ml834, full, L: ml835, full, y: ml838, full, tmp25: ml842, full
    ml843 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml842, 0.0, ml843)

    # R: ml834, full, L: ml835, full, y: ml838, full, tmp19: ml843, symmetric_lower_triangular
    ml844 = diag(ml835)
    ml845 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml834, 1, ml845, 1)
    # tmp29 = (L R)
    for i = 1:size(ml834, 2);
        view(ml834, :, i)[:] .*= ml844;
    end;        

    # R: ml845, full, y: ml838, full, tmp19: ml843, symmetric_lower_triangular, tmp29: ml834, full
    for i = 1:2000-1;
        view(ml843, i, i+1:2000)[:] = view(ml843, i+1:2000, i);
    end;
    ml846 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml843, 1, ml846, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml834, ml845, 1.0, ml843)

    # y: ml838, full, tmp19: ml846, full, tmp31: ml843, full
    ml847 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml843, ml847, info) = LinearAlgebra.LAPACK.getrf!(ml843)

    # y: ml838, full, tmp19: ml846, full, P35: ml847, ipiv, L33: ml843, lower_triangular_udiag, U34: ml843, upper_triangular
    ml848 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml846, ml838, 0.0, ml848)

    # P35: ml847, ipiv, L33: ml843, lower_triangular_udiag, U34: ml843, upper_triangular, tmp32: ml848, full
    ml849 = [1:length(ml847);]
    @inbounds for i in 1:length(ml847)
        ml849[i], ml849[ml847[i]] = ml849[ml847[i]], ml849[i];
    end;
    ml850 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml850 = ml848[ml849]

    # L33: ml843, lower_triangular_udiag, U34: ml843, upper_triangular, tmp40: ml850, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml843, ml850)

    # U34: ml843, upper_triangular, tmp41: ml850, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml843, ml850)

    # tmp17: ml850, full
    # x = tmp17
    return (ml850)
end