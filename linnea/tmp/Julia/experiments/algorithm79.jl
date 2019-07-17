using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm79(ml2663::Array{Float64,2}, ml2664::Array{Float64,2}, ml2665::Array{Float64,2}, ml2666::Array{Float64,2}, ml2667::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2663, full, L: ml2664, full, A: ml2665, full, B: ml2666, full, y: ml2667, full
    ml2668 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2668, ml2666)

    # R: ml2663, full, L: ml2664, full, A: ml2665, full, y: ml2667, full, tmp26: ml2668, full
    ml2669 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2665, ml2669, info) = LinearAlgebra.LAPACK.getrf!(ml2665)

    # R: ml2663, full, L: ml2664, full, y: ml2667, full, tmp26: ml2668, full, P11: ml2669, ipiv, L9: ml2665, lower_triangular_udiag, U10: ml2665, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2665, ml2668)

    # R: ml2663, full, L: ml2664, full, y: ml2667, full, P11: ml2669, ipiv, L9: ml2665, lower_triangular_udiag, tmp27: ml2668, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2665, ml2668)

    # R: ml2663, full, L: ml2664, full, y: ml2667, full, P11: ml2669, ipiv, tmp28: ml2668, full
    ml2670 = [1:length(ml2669);]
    @inbounds for i in 1:length(ml2669)
        ml2670[i], ml2670[ml2669[i]] = ml2670[ml2669[i]], ml2670[i];
    end;
    ml2671 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2671 = ml2668[invperm(ml2670),:]

    # R: ml2663, full, L: ml2664, full, y: ml2667, full, tmp25: ml2671, full
    ml2672 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2671, 0.0, ml2672)

    # R: ml2663, full, L: ml2664, full, y: ml2667, full, tmp19: ml2672, symmetric_lower_triangular
    ml2673 = diag(ml2664)
    ml2674 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2663, 1, ml2674, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2663, 2);
        view(ml2663, :, i)[:] .*= ml2673;
    end;        

    # R: ml2674, full, y: ml2667, full, tmp19: ml2672, symmetric_lower_triangular, tmp29: ml2663, full
    for i = 1:2000-1;
        view(ml2672, i, i+1:2000)[:] = view(ml2672, i+1:2000, i);
    end;
    ml2675 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2672, 1, ml2675, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2663, ml2674, 1.0, ml2672)

    # y: ml2667, full, tmp19: ml2675, full, tmp31: ml2672, full
    ml2676 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2672, ml2676, info) = LinearAlgebra.LAPACK.getrf!(ml2672)

    # y: ml2667, full, tmp19: ml2675, full, P35: ml2676, ipiv, L33: ml2672, lower_triangular_udiag, U34: ml2672, upper_triangular
    ml2677 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2675, ml2667, 0.0, ml2677)

    # P35: ml2676, ipiv, L33: ml2672, lower_triangular_udiag, U34: ml2672, upper_triangular, tmp32: ml2677, full
    ml2678 = [1:length(ml2676);]
    @inbounds for i in 1:length(ml2676)
        ml2678[i], ml2678[ml2676[i]] = ml2678[ml2676[i]], ml2678[i];
    end;
    ml2679 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2679 = ml2677[ml2678]

    # L33: ml2672, lower_triangular_udiag, U34: ml2672, upper_triangular, tmp40: ml2679, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2672, ml2679)

    # U34: ml2672, upper_triangular, tmp41: ml2679, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2672, ml2679)

    # tmp17: ml2679, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2679), (finish-start)*1e-9)
end