using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm25(ml851::Array{Float64,2}, ml852::Array{Float64,2}, ml853::Array{Float64,2}, ml854::Array{Float64,2}, ml855::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml851, full, L: ml852, full, A: ml853, full, B: ml854, full, y: ml855, full
    ml856 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml856, ml854)

    # R: ml851, full, L: ml852, full, A: ml853, full, y: ml855, full, tmp26: ml856, full
    ml857 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml853, ml857, info) = LinearAlgebra.LAPACK.getrf!(ml853)

    # R: ml851, full, L: ml852, full, y: ml855, full, tmp26: ml856, full, P11: ml857, ipiv, L9: ml853, lower_triangular_udiag, U10: ml853, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml853, ml856)

    # R: ml851, full, L: ml852, full, y: ml855, full, P11: ml857, ipiv, L9: ml853, lower_triangular_udiag, tmp27: ml856, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml853, ml856)

    # R: ml851, full, L: ml852, full, y: ml855, full, P11: ml857, ipiv, tmp28: ml856, full
    ml858 = [1:length(ml857);]
    @inbounds for i in 1:length(ml857)
        ml858[i], ml858[ml857[i]] = ml858[ml857[i]], ml858[i];
    end;
    ml859 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml859 = ml856[invperm(ml858),:]

    # R: ml851, full, L: ml852, full, y: ml855, full, tmp25: ml859, full
    ml860 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml859, 0.0, ml860)

    # R: ml851, full, L: ml852, full, y: ml855, full, tmp19: ml860, symmetric_lower_triangular
    ml861 = diag(ml852)
    ml862 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml851, 1, ml862, 1)
    # tmp29 = (L R)
    for i = 1:size(ml851, 2);
        view(ml851, :, i)[:] .*= ml861;
    end;        

    # R: ml862, full, y: ml855, full, tmp19: ml860, symmetric_lower_triangular, tmp29: ml851, full
    for i = 1:2000-1;
        view(ml860, i, i+1:2000)[:] = view(ml860, i+1:2000, i);
    end;
    ml863 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml860, 1, ml863, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml851, ml862, 1.0, ml860)

    # y: ml855, full, tmp19: ml863, full, tmp31: ml860, full
    ml864 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml860, ml864, info) = LinearAlgebra.LAPACK.getrf!(ml860)

    # y: ml855, full, tmp19: ml863, full, P35: ml864, ipiv, L33: ml860, lower_triangular_udiag, U34: ml860, upper_triangular
    ml865 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml863, ml855, 0.0, ml865)

    # P35: ml864, ipiv, L33: ml860, lower_triangular_udiag, U34: ml860, upper_triangular, tmp32: ml865, full
    ml866 = [1:length(ml864);]
    @inbounds for i in 1:length(ml864)
        ml866[i], ml866[ml864[i]] = ml866[ml864[i]], ml866[i];
    end;
    ml867 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml867 = ml865[ml866]

    # L33: ml860, lower_triangular_udiag, U34: ml860, upper_triangular, tmp40: ml867, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml860, ml867)

    # U34: ml860, upper_triangular, tmp41: ml867, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml860, ml867)

    # tmp17: ml867, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml867), (finish-start)*1e-9)
end