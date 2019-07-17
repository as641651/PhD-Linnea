using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm86(ml2893::Array{Float64,2}, ml2894::Array{Float64,2}, ml2895::Array{Float64,2}, ml2896::Array{Float64,2}, ml2897::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2893, full, L: ml2894, full, A: ml2895, full, B: ml2896, full, y: ml2897, full
    ml2898 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2898, ml2896)

    # R: ml2893, full, L: ml2894, full, A: ml2895, full, y: ml2897, full, tmp26: ml2898, full
    ml2899 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2895, ml2899, info) = LinearAlgebra.LAPACK.getrf!(ml2895)

    # R: ml2893, full, L: ml2894, full, y: ml2897, full, tmp26: ml2898, full, P11: ml2899, ipiv, L9: ml2895, lower_triangular_udiag, U10: ml2895, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2895, ml2898)

    # R: ml2893, full, L: ml2894, full, y: ml2897, full, P11: ml2899, ipiv, L9: ml2895, lower_triangular_udiag, tmp27: ml2898, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2895, ml2898)

    # R: ml2893, full, L: ml2894, full, y: ml2897, full, P11: ml2899, ipiv, tmp28: ml2898, full
    ml2900 = [1:length(ml2899);]
    @inbounds for i in 1:length(ml2899)
        ml2900[i], ml2900[ml2899[i]] = ml2900[ml2899[i]], ml2900[i];
    end;
    ml2901 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2901 = ml2898[invperm(ml2900),:]

    # R: ml2893, full, L: ml2894, full, y: ml2897, full, tmp25: ml2901, full
    ml2902 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2901, 0.0, ml2902)

    # R: ml2893, full, L: ml2894, full, y: ml2897, full, tmp19: ml2902, symmetric_lower_triangular
    ml2903 = diag(ml2894)
    ml2904 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2893, 1, ml2904, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2893, 2);
        view(ml2893, :, i)[:] .*= ml2903;
    end;        

    # R: ml2904, full, y: ml2897, full, tmp19: ml2902, symmetric_lower_triangular, tmp29: ml2893, full
    for i = 1:2000-1;
        view(ml2902, i, i+1:2000)[:] = view(ml2902, i+1:2000, i);
    end;
    ml2905 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2902, 1, ml2905, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2893, ml2904, 1.0, ml2902)

    # y: ml2897, full, tmp19: ml2905, full, tmp31: ml2902, full
    ml2906 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2902, ml2906, info) = LinearAlgebra.LAPACK.getrf!(ml2902)

    # y: ml2897, full, tmp19: ml2905, full, P35: ml2906, ipiv, L33: ml2902, lower_triangular_udiag, U34: ml2902, upper_triangular
    ml2907 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2905, ml2897, 0.0, ml2907)

    # P35: ml2906, ipiv, L33: ml2902, lower_triangular_udiag, U34: ml2902, upper_triangular, tmp32: ml2907, full
    ml2908 = [1:length(ml2906);]
    @inbounds for i in 1:length(ml2906)
        ml2908[i], ml2908[ml2906[i]] = ml2908[ml2906[i]], ml2908[i];
    end;
    ml2909 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2909 = ml2907[ml2908]

    # L33: ml2902, lower_triangular_udiag, U34: ml2902, upper_triangular, tmp40: ml2909, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2902, ml2909)

    # U34: ml2902, upper_triangular, tmp41: ml2909, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2902, ml2909)

    # tmp17: ml2909, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2909), (finish-start)*1e-9)
end