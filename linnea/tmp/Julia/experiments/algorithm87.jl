using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm87(ml2927::Array{Float64,2}, ml2928::Array{Float64,2}, ml2929::Array{Float64,2}, ml2930::Array{Float64,2}, ml2931::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2927, full, L: ml2928, full, A: ml2929, full, B: ml2930, full, y: ml2931, full
    ml2932 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2932, ml2930)

    # R: ml2927, full, L: ml2928, full, A: ml2929, full, y: ml2931, full, tmp26: ml2932, full
    ml2933 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2929, ml2933, info) = LinearAlgebra.LAPACK.getrf!(ml2929)

    # R: ml2927, full, L: ml2928, full, y: ml2931, full, tmp26: ml2932, full, P11: ml2933, ipiv, L9: ml2929, lower_triangular_udiag, U10: ml2929, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2929, ml2932)

    # R: ml2927, full, L: ml2928, full, y: ml2931, full, P11: ml2933, ipiv, L9: ml2929, lower_triangular_udiag, tmp27: ml2932, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2929, ml2932)

    # R: ml2927, full, L: ml2928, full, y: ml2931, full, P11: ml2933, ipiv, tmp28: ml2932, full
    ml2934 = [1:length(ml2933);]
    @inbounds for i in 1:length(ml2933)
        ml2934[i], ml2934[ml2933[i]] = ml2934[ml2933[i]], ml2934[i];
    end;
    ml2935 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2935 = ml2932[invperm(ml2934),:]

    # R: ml2927, full, L: ml2928, full, y: ml2931, full, tmp25: ml2935, full
    ml2936 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2935, 0.0, ml2936)

    # R: ml2927, full, L: ml2928, full, y: ml2931, full, tmp19: ml2936, symmetric_lower_triangular
    ml2937 = diag(ml2928)
    ml2938 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2927, 1, ml2938, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2927, 2);
        view(ml2927, :, i)[:] .*= ml2937;
    end;        

    # R: ml2938, full, y: ml2931, full, tmp19: ml2936, symmetric_lower_triangular, tmp29: ml2927, full
    for i = 1:2000-1;
        view(ml2936, i, i+1:2000)[:] = view(ml2936, i+1:2000, i);
    end;
    ml2939 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2936, 1, ml2939, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2927, ml2938, 1.0, ml2936)

    # y: ml2931, full, tmp19: ml2939, full, tmp31: ml2936, full
    ml2940 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2936, ml2940, info) = LinearAlgebra.LAPACK.getrf!(ml2936)

    # y: ml2931, full, tmp19: ml2939, full, P35: ml2940, ipiv, L33: ml2936, lower_triangular_udiag, U34: ml2936, upper_triangular
    ml2941 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2939, ml2931, 0.0, ml2941)

    # P35: ml2940, ipiv, L33: ml2936, lower_triangular_udiag, U34: ml2936, upper_triangular, tmp32: ml2941, full
    ml2942 = [1:length(ml2940);]
    @inbounds for i in 1:length(ml2940)
        ml2942[i], ml2942[ml2940[i]] = ml2942[ml2940[i]], ml2942[i];
    end;
    ml2943 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2943 = ml2941[ml2942]

    # L33: ml2936, lower_triangular_udiag, U34: ml2936, upper_triangular, tmp40: ml2943, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2936, ml2943)

    # U34: ml2936, upper_triangular, tmp41: ml2943, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2936, ml2943)

    # tmp17: ml2943, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2943), (finish-start)*1e-9)
end