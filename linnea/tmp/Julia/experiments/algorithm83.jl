using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm83(ml2792::Array{Float64,2}, ml2793::Array{Float64,2}, ml2794::Array{Float64,2}, ml2795::Array{Float64,2}, ml2796::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2792, full, L: ml2793, full, A: ml2794, full, B: ml2795, full, y: ml2796, full
    ml2797 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2797, ml2795)

    # R: ml2792, full, L: ml2793, full, A: ml2794, full, y: ml2796, full, tmp26: ml2797, full
    ml2798 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2794, ml2798, info) = LinearAlgebra.LAPACK.getrf!(ml2794)

    # R: ml2792, full, L: ml2793, full, y: ml2796, full, tmp26: ml2797, full, P11: ml2798, ipiv, L9: ml2794, lower_triangular_udiag, U10: ml2794, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2794, ml2797)

    # R: ml2792, full, L: ml2793, full, y: ml2796, full, P11: ml2798, ipiv, L9: ml2794, lower_triangular_udiag, tmp27: ml2797, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2794, ml2797)

    # R: ml2792, full, L: ml2793, full, y: ml2796, full, P11: ml2798, ipiv, tmp28: ml2797, full
    ml2799 = [1:length(ml2798);]
    @inbounds for i in 1:length(ml2798)
        ml2799[i], ml2799[ml2798[i]] = ml2799[ml2798[i]], ml2799[i];
    end;
    ml2800 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2800 = ml2797[invperm(ml2799),:]

    # R: ml2792, full, L: ml2793, full, y: ml2796, full, tmp25: ml2800, full
    ml2801 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2800, 0.0, ml2801)

    # R: ml2792, full, L: ml2793, full, y: ml2796, full, tmp19: ml2801, symmetric_lower_triangular
    ml2802 = diag(ml2793)
    ml2803 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2792, 1, ml2803, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2792, 2);
        view(ml2792, :, i)[:] .*= ml2802;
    end;        

    # R: ml2803, full, y: ml2796, full, tmp19: ml2801, symmetric_lower_triangular, tmp29: ml2792, full
    ml2804 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2801, ml2796, 0.0, ml2804)

    # R: ml2803, full, tmp19: ml2801, symmetric_lower_triangular, tmp29: ml2792, full, tmp32: ml2804, full
    for i = 1:2000-1;
        view(ml2801, i, i+1:2000)[:] = view(ml2801, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2792, ml2803, 1.0, ml2801)

    # tmp32: ml2804, full, tmp31: ml2801, full
    ml2805 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2801, ml2805, info) = LinearAlgebra.LAPACK.getrf!(ml2801)

    # tmp32: ml2804, full, P35: ml2805, ipiv, L33: ml2801, lower_triangular_udiag, U34: ml2801, upper_triangular
    ml2806 = [1:length(ml2805);]
    @inbounds for i in 1:length(ml2805)
        ml2806[i], ml2806[ml2805[i]] = ml2806[ml2805[i]], ml2806[i];
    end;
    ml2807 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2807 = ml2804[ml2806]

    # L33: ml2801, lower_triangular_udiag, U34: ml2801, upper_triangular, tmp40: ml2807, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2801, ml2807)

    # U34: ml2801, upper_triangular, tmp41: ml2807, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2801, ml2807)

    # tmp17: ml2807, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2807), (finish-start)*1e-9)
end