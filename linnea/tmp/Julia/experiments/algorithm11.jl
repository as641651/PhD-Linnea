using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm11(ml383::Array{Float64,2}, ml384::Array{Float64,2}, ml385::Array{Float64,2}, ml386::Array{Float64,2}, ml387::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml383, full, L: ml384, full, A: ml385, full, B: ml386, full, y: ml387, full
    ml388 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml385, ml388, info) = LinearAlgebra.LAPACK.getrf!(ml385)

    # R: ml383, full, L: ml384, full, B: ml386, full, y: ml387, full, P11: ml388, ipiv, L9: ml385, lower_triangular_udiag, U10: ml385, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml385, ml386)

    # R: ml383, full, L: ml384, full, y: ml387, full, P11: ml388, ipiv, L9: ml385, lower_triangular_udiag, tmp53: ml386, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml385, ml386)

    # R: ml383, full, L: ml384, full, y: ml387, full, P11: ml388, ipiv, tmp54: ml386, full
    ml389 = [1:length(ml388);]
    @inbounds for i in 1:length(ml388)
        ml389[i], ml389[ml388[i]] = ml389[ml388[i]], ml389[i];
    end;
    ml390 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml390 = ml386[:,invperm(ml389)]

    # R: ml383, full, L: ml384, full, y: ml387, full, tmp55: ml390, full
    ml391 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml391, ml390)

    # R: ml383, full, L: ml384, full, y: ml387, full, tmp25: ml391, full
    ml392 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml391, 0.0, ml392)

    # R: ml383, full, L: ml384, full, y: ml387, full, tmp19: ml392, symmetric_lower_triangular
    ml393 = diag(ml384)
    ml394 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml383, 1, ml394, 1)
    # tmp29 = (L R)
    for i = 1:size(ml383, 2);
        view(ml383, :, i)[:] .*= ml393;
    end;        

    # R: ml394, full, y: ml387, full, tmp19: ml392, symmetric_lower_triangular, tmp29: ml383, full
    for i = 1:2000-1;
        view(ml392, i, i+1:2000)[:] = view(ml392, i+1:2000, i);
    end;
    ml395 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml392, 1, ml395, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml383, ml394, 1.0, ml392)

    # y: ml387, full, tmp19: ml395, full, tmp31: ml392, full
    ml396 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml395, ml387, 0.0, ml396)

    # tmp31: ml392, full, tmp32: ml396, full
    ml397 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml392, ml397, info) = LinearAlgebra.LAPACK.getrf!(ml392)

    # tmp32: ml396, full, P35: ml397, ipiv, L33: ml392, lower_triangular_udiag, U34: ml392, upper_triangular
    ml398 = [1:length(ml397);]
    @inbounds for i in 1:length(ml397)
        ml398[i], ml398[ml397[i]] = ml398[ml397[i]], ml398[i];
    end;
    ml399 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml399 = ml396[ml398]

    # L33: ml392, lower_triangular_udiag, U34: ml392, upper_triangular, tmp40: ml399, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml392, ml399)

    # U34: ml392, upper_triangular, tmp41: ml399, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml392, ml399)

    # tmp17: ml399, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml399), (finish-start)*1e-9)
end