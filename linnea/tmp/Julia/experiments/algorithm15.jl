using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm15(ml519::Array{Float64,2}, ml520::Array{Float64,2}, ml521::Array{Float64,2}, ml522::Array{Float64,2}, ml523::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml519, full, L: ml520, full, A: ml521, full, B: ml522, full, y: ml523, full
    ml524 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml524, ml522)

    # R: ml519, full, L: ml520, full, A: ml521, full, y: ml523, full, tmp26: ml524, full
    ml525 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml521, ml525, info) = LinearAlgebra.LAPACK.getrf!(ml521)

    # R: ml519, full, L: ml520, full, y: ml523, full, tmp26: ml524, full, P11: ml525, ipiv, L9: ml521, lower_triangular_udiag, U10: ml521, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml521, ml524)

    # R: ml519, full, L: ml520, full, y: ml523, full, P11: ml525, ipiv, L9: ml521, lower_triangular_udiag, tmp27: ml524, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml521, ml524)

    # R: ml519, full, L: ml520, full, y: ml523, full, P11: ml525, ipiv, tmp28: ml524, full
    ml526 = [1:length(ml525);]
    @inbounds for i in 1:length(ml525)
        ml526[i], ml526[ml525[i]] = ml526[ml525[i]], ml526[i];
    end;
    ml527 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml527 = ml524[invperm(ml526),:]

    # R: ml519, full, L: ml520, full, y: ml523, full, tmp25: ml527, full
    ml528 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml527, 0.0, ml528)

    # R: ml519, full, L: ml520, full, y: ml523, full, tmp19: ml528, symmetric_lower_triangular
    ml529 = diag(ml520)
    ml530 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml519, 1, ml530, 1)
    # tmp29 = (L R)
    for i = 1:size(ml519, 2);
        view(ml519, :, i)[:] .*= ml529;
    end;        

    # R: ml530, full, y: ml523, full, tmp19: ml528, symmetric_lower_triangular, tmp29: ml519, full
    for i = 1:2000-1;
        view(ml528, i, i+1:2000)[:] = view(ml528, i+1:2000, i);
    end;
    ml531 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml528, 1, ml531, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml519, ml530, 1.0, ml528)

    # y: ml523, full, tmp19: ml531, full, tmp31: ml528, full
    ml532 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml528, ml532, info) = LinearAlgebra.LAPACK.getrf!(ml528)

    # y: ml523, full, tmp19: ml531, full, P35: ml532, ipiv, L33: ml528, lower_triangular_udiag, U34: ml528, upper_triangular
    ml533 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml531, ml523, 0.0, ml533)

    # P35: ml532, ipiv, L33: ml528, lower_triangular_udiag, U34: ml528, upper_triangular, tmp32: ml533, full
    ml534 = [1:length(ml532);]
    @inbounds for i in 1:length(ml532)
        ml534[i], ml534[ml532[i]] = ml534[ml532[i]], ml534[i];
    end;
    ml535 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml535 = ml533[ml534]

    # L33: ml528, lower_triangular_udiag, U34: ml528, upper_triangular, tmp40: ml535, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml528, ml535)

    # U34: ml528, upper_triangular, tmp41: ml535, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml528, ml535)

    # tmp17: ml535, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml535), (finish-start)*1e-9)
end