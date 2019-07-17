using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm5(ml179::Array{Float64,2}, ml180::Array{Float64,2}, ml181::Array{Float64,2}, ml182::Array{Float64,2}, ml183::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml179, full, L: ml180, full, A: ml181, full, B: ml182, full, y: ml183, full
    ml184 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml181, ml184, info) = LinearAlgebra.LAPACK.getrf!(ml181)

    # R: ml179, full, L: ml180, full, B: ml182, full, y: ml183, full, P11: ml184, ipiv, L9: ml181, lower_triangular_udiag, U10: ml181, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml181, ml182)

    # R: ml179, full, L: ml180, full, y: ml183, full, P11: ml184, ipiv, L9: ml181, lower_triangular_udiag, tmp53: ml182, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml181, ml182)

    # R: ml179, full, L: ml180, full, y: ml183, full, P11: ml184, ipiv, tmp54: ml182, full
    ml185 = [1:length(ml184);]
    @inbounds for i in 1:length(ml184)
        ml185[i], ml185[ml184[i]] = ml185[ml184[i]], ml185[i];
    end;
    ml186 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml186 = ml182[:,invperm(ml185)]

    # R: ml179, full, L: ml180, full, y: ml183, full, tmp55: ml186, full
    ml187 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml187, ml186)

    # R: ml179, full, L: ml180, full, y: ml183, full, tmp25: ml187, full
    ml188 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml187, 0.0, ml188)

    # R: ml179, full, L: ml180, full, y: ml183, full, tmp19: ml188, symmetric_lower_triangular
    ml189 = diag(ml180)
    ml190 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml179, 1, ml190, 1)
    # tmp29 = (L R)
    for i = 1:size(ml179, 2);
        view(ml179, :, i)[:] .*= ml189;
    end;        

    # R: ml190, full, y: ml183, full, tmp19: ml188, symmetric_lower_triangular, tmp29: ml179, full
    for i = 1:2000-1;
        view(ml188, i, i+1:2000)[:] = view(ml188, i+1:2000, i);
    end;
    ml191 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml188, 1, ml191, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml179, ml190, 1.0, ml188)

    # y: ml183, full, tmp19: ml191, full, tmp31: ml188, full
    ml192 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml188, ml192, info) = LinearAlgebra.LAPACK.getrf!(ml188)

    # y: ml183, full, tmp19: ml191, full, P35: ml192, ipiv, L33: ml188, lower_triangular_udiag, U34: ml188, upper_triangular
    ml193 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml191, ml183, 0.0, ml193)

    # P35: ml192, ipiv, L33: ml188, lower_triangular_udiag, U34: ml188, upper_triangular, tmp32: ml193, full
    ml194 = [1:length(ml192);]
    @inbounds for i in 1:length(ml192)
        ml194[i], ml194[ml192[i]] = ml194[ml192[i]], ml194[i];
    end;
    ml195 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml195 = ml193[ml194]

    # L33: ml188, lower_triangular_udiag, U34: ml188, upper_triangular, tmp40: ml195, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml188, ml195)

    # U34: ml188, upper_triangular, tmp41: ml195, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml188, ml195)

    # tmp17: ml195, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml195), (finish-start)*1e-9)
end