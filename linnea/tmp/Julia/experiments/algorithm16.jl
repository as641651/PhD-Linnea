using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm16(ml553::Array{Float64,2}, ml554::Array{Float64,2}, ml555::Array{Float64,2}, ml556::Array{Float64,2}, ml557::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml553, full, L: ml554, full, A: ml555, full, B: ml556, full, y: ml557, full
    ml558 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml558, ml556)

    # R: ml553, full, L: ml554, full, A: ml555, full, y: ml557, full, tmp26: ml558, full
    ml559 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml555, ml559, info) = LinearAlgebra.LAPACK.getrf!(ml555)

    # R: ml553, full, L: ml554, full, y: ml557, full, tmp26: ml558, full, P11: ml559, ipiv, L9: ml555, lower_triangular_udiag, U10: ml555, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml555, ml558)

    # R: ml553, full, L: ml554, full, y: ml557, full, P11: ml559, ipiv, L9: ml555, lower_triangular_udiag, tmp27: ml558, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml555, ml558)

    # R: ml553, full, L: ml554, full, y: ml557, full, P11: ml559, ipiv, tmp28: ml558, full
    ml560 = [1:length(ml559);]
    @inbounds for i in 1:length(ml559)
        ml560[i], ml560[ml559[i]] = ml560[ml559[i]], ml560[i];
    end;
    ml561 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml561 = ml558[invperm(ml560),:]

    # R: ml553, full, L: ml554, full, y: ml557, full, tmp25: ml561, full
    ml562 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml561, 0.0, ml562)

    # R: ml553, full, L: ml554, full, y: ml557, full, tmp19: ml562, symmetric_lower_triangular
    ml563 = diag(ml554)
    ml564 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml553, 1, ml564, 1)
    # tmp29 = (L R)
    for i = 1:size(ml553, 2);
        view(ml553, :, i)[:] .*= ml563;
    end;        

    # R: ml564, full, y: ml557, full, tmp19: ml562, symmetric_lower_triangular, tmp29: ml553, full
    for i = 1:2000-1;
        view(ml562, i, i+1:2000)[:] = view(ml562, i+1:2000, i);
    end;
    ml565 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml562, 1, ml565, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml553, ml564, 1.0, ml562)

    # y: ml557, full, tmp19: ml565, full, tmp31: ml562, full
    ml566 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml562, ml566, info) = LinearAlgebra.LAPACK.getrf!(ml562)

    # y: ml557, full, tmp19: ml565, full, P35: ml566, ipiv, L33: ml562, lower_triangular_udiag, U34: ml562, upper_triangular
    ml567 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml565, ml557, 0.0, ml567)

    # P35: ml566, ipiv, L33: ml562, lower_triangular_udiag, U34: ml562, upper_triangular, tmp32: ml567, full
    ml568 = [1:length(ml566);]
    @inbounds for i in 1:length(ml566)
        ml568[i], ml568[ml566[i]] = ml568[ml566[i]], ml568[i];
    end;
    ml569 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml569 = ml567[ml568]

    # L33: ml562, lower_triangular_udiag, U34: ml562, upper_triangular, tmp40: ml569, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml562, ml569)

    # U34: ml562, upper_triangular, tmp41: ml569, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml562, ml569)

    # tmp17: ml569, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml569), (finish-start)*1e-9)
end