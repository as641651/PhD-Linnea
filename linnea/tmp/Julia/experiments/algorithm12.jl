using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm12(ml417::Array{Float64,2}, ml418::Array{Float64,2}, ml419::Array{Float64,2}, ml420::Array{Float64,2}, ml421::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml417, full, L: ml418, full, A: ml419, full, B: ml420, full, y: ml421, full
    ml422 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml419, ml422, info) = LinearAlgebra.LAPACK.getrf!(ml419)

    # R: ml417, full, L: ml418, full, B: ml420, full, y: ml421, full, P11: ml422, ipiv, L9: ml419, lower_triangular_udiag, U10: ml419, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml419, ml420)

    # R: ml417, full, L: ml418, full, y: ml421, full, P11: ml422, ipiv, L9: ml419, lower_triangular_udiag, tmp53: ml420, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml419, ml420)

    # R: ml417, full, L: ml418, full, y: ml421, full, P11: ml422, ipiv, tmp54: ml420, full
    ml423 = [1:length(ml422);]
    @inbounds for i in 1:length(ml422)
        ml423[i], ml423[ml422[i]] = ml423[ml422[i]], ml423[i];
    end;
    ml424 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml424 = ml420[:,invperm(ml423)]

    # R: ml417, full, L: ml418, full, y: ml421, full, tmp55: ml424, full
    ml425 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml425, ml424)

    # R: ml417, full, L: ml418, full, y: ml421, full, tmp25: ml425, full
    ml426 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml425, 0.0, ml426)

    # R: ml417, full, L: ml418, full, y: ml421, full, tmp19: ml426, symmetric_lower_triangular
    ml427 = diag(ml418)
    ml428 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml417, 1, ml428, 1)
    # tmp29 = (L R)
    for i = 1:size(ml417, 2);
        view(ml417, :, i)[:] .*= ml427;
    end;        

    # R: ml428, full, y: ml421, full, tmp19: ml426, symmetric_lower_triangular, tmp29: ml417, full
    for i = 1:2000-1;
        view(ml426, i, i+1:2000)[:] = view(ml426, i+1:2000, i);
    end;
    ml429 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml426, 1, ml429, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml417, ml428, 1.0, ml426)

    # y: ml421, full, tmp19: ml429, full, tmp31: ml426, full
    ml430 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml429, ml421, 0.0, ml430)

    # tmp31: ml426, full, tmp32: ml430, full
    ml431 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml426, ml431, info) = LinearAlgebra.LAPACK.getrf!(ml426)

    # tmp32: ml430, full, P35: ml431, ipiv, L33: ml426, lower_triangular_udiag, U34: ml426, upper_triangular
    ml432 = [1:length(ml431);]
    @inbounds for i in 1:length(ml431)
        ml432[i], ml432[ml431[i]] = ml432[ml431[i]], ml432[i];
    end;
    ml433 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml433 = ml430[ml432]

    # L33: ml426, lower_triangular_udiag, U34: ml426, upper_triangular, tmp40: ml433, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml426, ml433)

    # U34: ml426, upper_triangular, tmp41: ml433, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml426, ml433)

    # tmp17: ml433, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml433), (finish-start)*1e-9)
end