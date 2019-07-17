using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm92(ml3092::Array{Float64,2}, ml3093::Array{Float64,2}, ml3094::Array{Float64,2}, ml3095::Array{Float64,2}, ml3096::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3092, full, L: ml3093, full, A: ml3094, full, B: ml3095, full, y: ml3096, full
    ml3097 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3097, ml3095)

    # R: ml3092, full, L: ml3093, full, A: ml3094, full, y: ml3096, full, tmp26: ml3097, full
    ml3098 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3094, ml3098, info) = LinearAlgebra.LAPACK.getrf!(ml3094)

    # R: ml3092, full, L: ml3093, full, y: ml3096, full, tmp26: ml3097, full, P11: ml3098, ipiv, L9: ml3094, lower_triangular_udiag, U10: ml3094, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3094, ml3097)

    # R: ml3092, full, L: ml3093, full, y: ml3096, full, P11: ml3098, ipiv, L9: ml3094, lower_triangular_udiag, tmp27: ml3097, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3094, ml3097)

    # R: ml3092, full, L: ml3093, full, y: ml3096, full, P11: ml3098, ipiv, tmp28: ml3097, full
    ml3099 = [1:length(ml3098);]
    @inbounds for i in 1:length(ml3098)
        ml3099[i], ml3099[ml3098[i]] = ml3099[ml3098[i]], ml3099[i];
    end;
    ml3100 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3100 = ml3097[invperm(ml3099),:]

    # R: ml3092, full, L: ml3093, full, y: ml3096, full, tmp25: ml3100, full
    ml3101 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3100, 0.0, ml3101)

    # R: ml3092, full, L: ml3093, full, y: ml3096, full, tmp19: ml3101, symmetric_lower_triangular
    ml3102 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3101, ml3096, 0.0, ml3102)

    # R: ml3092, full, L: ml3093, full, tmp19: ml3101, symmetric_lower_triangular, tmp32: ml3102, full
    ml3103 = diag(ml3093)
    ml3104 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3092, 1, ml3104, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3092, 2);
        view(ml3092, :, i)[:] .*= ml3103;
    end;        

    # R: ml3104, full, tmp19: ml3101, symmetric_lower_triangular, tmp32: ml3102, full, tmp29: ml3092, full
    for i = 1:2000-1;
        view(ml3101, i, i+1:2000)[:] = view(ml3101, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3104, ml3092, 1.0, ml3101)

    # tmp32: ml3102, full, tmp31: ml3101, full
    ml3105 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3101, ml3105, info) = LinearAlgebra.LAPACK.getrf!(ml3101)

    # tmp32: ml3102, full, P35: ml3105, ipiv, L33: ml3101, lower_triangular_udiag, U34: ml3101, upper_triangular
    ml3106 = [1:length(ml3105);]
    @inbounds for i in 1:length(ml3105)
        ml3106[i], ml3106[ml3105[i]] = ml3106[ml3105[i]], ml3106[i];
    end;
    ml3107 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3107 = ml3102[ml3106]

    # L33: ml3101, lower_triangular_udiag, U34: ml3101, upper_triangular, tmp40: ml3107, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3101, ml3107)

    # U34: ml3101, upper_triangular, tmp41: ml3107, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3101, ml3107)

    # tmp17: ml3107, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3107), (finish-start)*1e-9)
end