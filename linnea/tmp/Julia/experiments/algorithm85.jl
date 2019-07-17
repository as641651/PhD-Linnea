using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm85(ml2859::Array{Float64,2}, ml2860::Array{Float64,2}, ml2861::Array{Float64,2}, ml2862::Array{Float64,2}, ml2863::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2859, full, L: ml2860, full, A: ml2861, full, B: ml2862, full, y: ml2863, full
    ml2864 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2864, ml2862)

    # R: ml2859, full, L: ml2860, full, A: ml2861, full, y: ml2863, full, tmp26: ml2864, full
    ml2865 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2861, ml2865, info) = LinearAlgebra.LAPACK.getrf!(ml2861)

    # R: ml2859, full, L: ml2860, full, y: ml2863, full, tmp26: ml2864, full, P11: ml2865, ipiv, L9: ml2861, lower_triangular_udiag, U10: ml2861, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2861, ml2864)

    # R: ml2859, full, L: ml2860, full, y: ml2863, full, P11: ml2865, ipiv, L9: ml2861, lower_triangular_udiag, tmp27: ml2864, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2861, ml2864)

    # R: ml2859, full, L: ml2860, full, y: ml2863, full, P11: ml2865, ipiv, tmp28: ml2864, full
    ml2866 = [1:length(ml2865);]
    @inbounds for i in 1:length(ml2865)
        ml2866[i], ml2866[ml2865[i]] = ml2866[ml2865[i]], ml2866[i];
    end;
    ml2867 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2867 = ml2864[invperm(ml2866),:]

    # R: ml2859, full, L: ml2860, full, y: ml2863, full, tmp25: ml2867, full
    ml2868 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2867, 0.0, ml2868)

    # R: ml2859, full, L: ml2860, full, y: ml2863, full, tmp19: ml2868, symmetric_lower_triangular
    ml2869 = diag(ml2860)
    ml2870 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2859, 1, ml2870, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2859, 2);
        view(ml2859, :, i)[:] .*= ml2869;
    end;        

    # R: ml2870, full, y: ml2863, full, tmp19: ml2868, symmetric_lower_triangular, tmp29: ml2859, full
    for i = 1:2000-1;
        view(ml2868, i, i+1:2000)[:] = view(ml2868, i+1:2000, i);
    end;
    ml2871 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2868, 1, ml2871, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2859, ml2870, 1.0, ml2868)

    # y: ml2863, full, tmp19: ml2871, full, tmp31: ml2868, full
    ml2872 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2868, ml2872, info) = LinearAlgebra.LAPACK.getrf!(ml2868)

    # y: ml2863, full, tmp19: ml2871, full, P35: ml2872, ipiv, L33: ml2868, lower_triangular_udiag, U34: ml2868, upper_triangular
    ml2873 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2871, ml2863, 0.0, ml2873)

    # P35: ml2872, ipiv, L33: ml2868, lower_triangular_udiag, U34: ml2868, upper_triangular, tmp32: ml2873, full
    ml2874 = [1:length(ml2872);]
    @inbounds for i in 1:length(ml2872)
        ml2874[i], ml2874[ml2872[i]] = ml2874[ml2872[i]], ml2874[i];
    end;
    ml2875 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2875 = ml2873[ml2874]

    # L33: ml2868, lower_triangular_udiag, U34: ml2868, upper_triangular, tmp40: ml2875, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2868, ml2875)

    # U34: ml2868, upper_triangular, tmp41: ml2875, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2868, ml2875)

    # tmp17: ml2875, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2875), (finish-start)*1e-9)
end