using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm84(ml2825::Array{Float64,2}, ml2826::Array{Float64,2}, ml2827::Array{Float64,2}, ml2828::Array{Float64,2}, ml2829::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2825, full, L: ml2826, full, A: ml2827, full, B: ml2828, full, y: ml2829, full
    ml2830 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2830, ml2828)

    # R: ml2825, full, L: ml2826, full, A: ml2827, full, y: ml2829, full, tmp26: ml2830, full
    ml2831 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2827, ml2831, info) = LinearAlgebra.LAPACK.getrf!(ml2827)

    # R: ml2825, full, L: ml2826, full, y: ml2829, full, tmp26: ml2830, full, P11: ml2831, ipiv, L9: ml2827, lower_triangular_udiag, U10: ml2827, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2827, ml2830)

    # R: ml2825, full, L: ml2826, full, y: ml2829, full, P11: ml2831, ipiv, L9: ml2827, lower_triangular_udiag, tmp27: ml2830, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2827, ml2830)

    # R: ml2825, full, L: ml2826, full, y: ml2829, full, P11: ml2831, ipiv, tmp28: ml2830, full
    ml2832 = [1:length(ml2831);]
    @inbounds for i in 1:length(ml2831)
        ml2832[i], ml2832[ml2831[i]] = ml2832[ml2831[i]], ml2832[i];
    end;
    ml2833 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2833 = ml2830[invperm(ml2832),:]

    # R: ml2825, full, L: ml2826, full, y: ml2829, full, tmp25: ml2833, full
    ml2834 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2833, 0.0, ml2834)

    # R: ml2825, full, L: ml2826, full, y: ml2829, full, tmp19: ml2834, symmetric_lower_triangular
    ml2835 = diag(ml2826)
    ml2836 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2825, 1, ml2836, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2825, 2);
        view(ml2825, :, i)[:] .*= ml2835;
    end;        

    # R: ml2836, full, y: ml2829, full, tmp19: ml2834, symmetric_lower_triangular, tmp29: ml2825, full
    for i = 1:2000-1;
        view(ml2834, i, i+1:2000)[:] = view(ml2834, i+1:2000, i);
    end;
    ml2837 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2834, 1, ml2837, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2825, ml2836, 1.0, ml2834)

    # y: ml2829, full, tmp19: ml2837, full, tmp31: ml2834, full
    ml2838 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2834, ml2838, info) = LinearAlgebra.LAPACK.getrf!(ml2834)

    # y: ml2829, full, tmp19: ml2837, full, P35: ml2838, ipiv, L33: ml2834, lower_triangular_udiag, U34: ml2834, upper_triangular
    ml2839 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2837, ml2829, 0.0, ml2839)

    # P35: ml2838, ipiv, L33: ml2834, lower_triangular_udiag, U34: ml2834, upper_triangular, tmp32: ml2839, full
    ml2840 = [1:length(ml2838);]
    @inbounds for i in 1:length(ml2838)
        ml2840[i], ml2840[ml2838[i]] = ml2840[ml2838[i]], ml2840[i];
    end;
    ml2841 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2841 = ml2839[ml2840]

    # L33: ml2834, lower_triangular_udiag, U34: ml2834, upper_triangular, tmp40: ml2841, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2834, ml2841)

    # U34: ml2834, upper_triangular, tmp41: ml2841, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2834, ml2841)

    # tmp17: ml2841, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2841), (finish-start)*1e-9)
end