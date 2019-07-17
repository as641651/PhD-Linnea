using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm88(ml2961::Array{Float64,2}, ml2962::Array{Float64,2}, ml2963::Array{Float64,2}, ml2964::Array{Float64,2}, ml2965::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2961, full, L: ml2962, full, A: ml2963, full, B: ml2964, full, y: ml2965, full
    ml2966 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2966, ml2964)

    # R: ml2961, full, L: ml2962, full, A: ml2963, full, y: ml2965, full, tmp26: ml2966, full
    ml2967 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2963, ml2967, info) = LinearAlgebra.LAPACK.getrf!(ml2963)

    # R: ml2961, full, L: ml2962, full, y: ml2965, full, tmp26: ml2966, full, P11: ml2967, ipiv, L9: ml2963, lower_triangular_udiag, U10: ml2963, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2963, ml2966)

    # R: ml2961, full, L: ml2962, full, y: ml2965, full, P11: ml2967, ipiv, L9: ml2963, lower_triangular_udiag, tmp27: ml2966, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2963, ml2966)

    # R: ml2961, full, L: ml2962, full, y: ml2965, full, P11: ml2967, ipiv, tmp28: ml2966, full
    ml2968 = [1:length(ml2967);]
    @inbounds for i in 1:length(ml2967)
        ml2968[i], ml2968[ml2967[i]] = ml2968[ml2967[i]], ml2968[i];
    end;
    ml2969 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2969 = ml2966[invperm(ml2968),:]

    # R: ml2961, full, L: ml2962, full, y: ml2965, full, tmp25: ml2969, full
    ml2970 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2969, 0.0, ml2970)

    # R: ml2961, full, L: ml2962, full, y: ml2965, full, tmp19: ml2970, symmetric_lower_triangular
    ml2971 = diag(ml2962)
    ml2972 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2961, 1, ml2972, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2961, 2);
        view(ml2961, :, i)[:] .*= ml2971;
    end;        

    # R: ml2972, full, y: ml2965, full, tmp19: ml2970, symmetric_lower_triangular, tmp29: ml2961, full
    for i = 1:2000-1;
        view(ml2970, i, i+1:2000)[:] = view(ml2970, i+1:2000, i);
    end;
    ml2973 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2970, 1, ml2973, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2961, ml2972, 1.0, ml2970)

    # y: ml2965, full, tmp19: ml2973, full, tmp31: ml2970, full
    ml2974 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2970, ml2974, info) = LinearAlgebra.LAPACK.getrf!(ml2970)

    # y: ml2965, full, tmp19: ml2973, full, P35: ml2974, ipiv, L33: ml2970, lower_triangular_udiag, U34: ml2970, upper_triangular
    ml2975 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2973, ml2965, 0.0, ml2975)

    # P35: ml2974, ipiv, L33: ml2970, lower_triangular_udiag, U34: ml2970, upper_triangular, tmp32: ml2975, full
    ml2976 = [1:length(ml2974);]
    @inbounds for i in 1:length(ml2974)
        ml2976[i], ml2976[ml2974[i]] = ml2976[ml2974[i]], ml2976[i];
    end;
    ml2977 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2977 = ml2975[ml2976]

    # L33: ml2970, lower_triangular_udiag, U34: ml2970, upper_triangular, tmp40: ml2977, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2970, ml2977)

    # U34: ml2970, upper_triangular, tmp41: ml2977, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2970, ml2977)

    # tmp17: ml2977, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2977), (finish-start)*1e-9)
end