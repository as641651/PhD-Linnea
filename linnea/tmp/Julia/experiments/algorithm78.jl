using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm78(ml2629::Array{Float64,2}, ml2630::Array{Float64,2}, ml2631::Array{Float64,2}, ml2632::Array{Float64,2}, ml2633::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2629, full, L: ml2630, full, A: ml2631, full, B: ml2632, full, y: ml2633, full
    ml2634 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2634, ml2632)

    # R: ml2629, full, L: ml2630, full, A: ml2631, full, y: ml2633, full, tmp26: ml2634, full
    ml2635 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2631, ml2635, info) = LinearAlgebra.LAPACK.getrf!(ml2631)

    # R: ml2629, full, L: ml2630, full, y: ml2633, full, tmp26: ml2634, full, P11: ml2635, ipiv, L9: ml2631, lower_triangular_udiag, U10: ml2631, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2631, ml2634)

    # R: ml2629, full, L: ml2630, full, y: ml2633, full, P11: ml2635, ipiv, L9: ml2631, lower_triangular_udiag, tmp27: ml2634, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2631, ml2634)

    # R: ml2629, full, L: ml2630, full, y: ml2633, full, P11: ml2635, ipiv, tmp28: ml2634, full
    ml2636 = [1:length(ml2635);]
    @inbounds for i in 1:length(ml2635)
        ml2636[i], ml2636[ml2635[i]] = ml2636[ml2635[i]], ml2636[i];
    end;
    ml2637 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2637 = ml2634[invperm(ml2636),:]

    # R: ml2629, full, L: ml2630, full, y: ml2633, full, tmp25: ml2637, full
    ml2638 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2637, 0.0, ml2638)

    # R: ml2629, full, L: ml2630, full, y: ml2633, full, tmp19: ml2638, symmetric_lower_triangular
    ml2639 = diag(ml2630)
    ml2640 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2629, 1, ml2640, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2629, 2);
        view(ml2629, :, i)[:] .*= ml2639;
    end;        

    # R: ml2640, full, y: ml2633, full, tmp19: ml2638, symmetric_lower_triangular, tmp29: ml2629, full
    for i = 1:2000-1;
        view(ml2638, i, i+1:2000)[:] = view(ml2638, i+1:2000, i);
    end;
    ml2641 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2638, 1, ml2641, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2629, ml2640, 1.0, ml2638)

    # y: ml2633, full, tmp19: ml2641, full, tmp31: ml2638, full
    ml2642 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2638, ml2642, info) = LinearAlgebra.LAPACK.getrf!(ml2638)

    # y: ml2633, full, tmp19: ml2641, full, P35: ml2642, ipiv, L33: ml2638, lower_triangular_udiag, U34: ml2638, upper_triangular
    ml2643 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2641, ml2633, 0.0, ml2643)

    # P35: ml2642, ipiv, L33: ml2638, lower_triangular_udiag, U34: ml2638, upper_triangular, tmp32: ml2643, full
    ml2644 = [1:length(ml2642);]
    @inbounds for i in 1:length(ml2642)
        ml2644[i], ml2644[ml2642[i]] = ml2644[ml2642[i]], ml2644[i];
    end;
    ml2645 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2645 = ml2643[ml2644]

    # L33: ml2638, lower_triangular_udiag, U34: ml2638, upper_triangular, tmp40: ml2645, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2638, ml2645)

    # U34: ml2638, upper_triangular, tmp41: ml2645, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2638, ml2645)

    # tmp17: ml2645, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2645), (finish-start)*1e-9)
end