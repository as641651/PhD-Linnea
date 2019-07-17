using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm93(ml3124::Array{Float64,2}, ml3125::Array{Float64,2}, ml3126::Array{Float64,2}, ml3127::Array{Float64,2}, ml3128::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3124, full, L: ml3125, full, A: ml3126, full, B: ml3127, full, y: ml3128, full
    ml3129 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3129, ml3127)

    # R: ml3124, full, L: ml3125, full, A: ml3126, full, y: ml3128, full, tmp26: ml3129, full
    ml3130 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3126, ml3130, info) = LinearAlgebra.LAPACK.getrf!(ml3126)

    # R: ml3124, full, L: ml3125, full, y: ml3128, full, tmp26: ml3129, full, P11: ml3130, ipiv, L9: ml3126, lower_triangular_udiag, U10: ml3126, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3126, ml3129)

    # R: ml3124, full, L: ml3125, full, y: ml3128, full, P11: ml3130, ipiv, L9: ml3126, lower_triangular_udiag, tmp27: ml3129, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3126, ml3129)

    # R: ml3124, full, L: ml3125, full, y: ml3128, full, P11: ml3130, ipiv, tmp28: ml3129, full
    ml3131 = [1:length(ml3130);]
    @inbounds for i in 1:length(ml3130)
        ml3131[i], ml3131[ml3130[i]] = ml3131[ml3130[i]], ml3131[i];
    end;
    ml3132 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3132 = ml3129[invperm(ml3131),:]

    # R: ml3124, full, L: ml3125, full, y: ml3128, full, tmp25: ml3132, full
    ml3133 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3132, 0.0, ml3133)

    # R: ml3124, full, L: ml3125, full, y: ml3128, full, tmp19: ml3133, symmetric_lower_triangular
    ml3134 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3133, ml3128, 0.0, ml3134)

    # R: ml3124, full, L: ml3125, full, tmp19: ml3133, symmetric_lower_triangular, tmp32: ml3134, full
    ml3135 = diag(ml3125)
    ml3136 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3124, 1, ml3136, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3124, 2);
        view(ml3124, :, i)[:] .*= ml3135;
    end;        

    # R: ml3136, full, tmp19: ml3133, symmetric_lower_triangular, tmp32: ml3134, full, tmp29: ml3124, full
    for i = 1:2000-1;
        view(ml3133, i, i+1:2000)[:] = view(ml3133, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3136, ml3124, 1.0, ml3133)

    # tmp32: ml3134, full, tmp31: ml3133, full
    ml3137 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3133, ml3137, info) = LinearAlgebra.LAPACK.getrf!(ml3133)

    # tmp32: ml3134, full, P35: ml3137, ipiv, L33: ml3133, lower_triangular_udiag, U34: ml3133, upper_triangular
    ml3138 = [1:length(ml3137);]
    @inbounds for i in 1:length(ml3137)
        ml3138[i], ml3138[ml3137[i]] = ml3138[ml3137[i]], ml3138[i];
    end;
    ml3139 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3139 = ml3134[ml3138]

    # L33: ml3133, lower_triangular_udiag, U34: ml3133, upper_triangular, tmp40: ml3139, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3133, ml3139)

    # U34: ml3133, upper_triangular, tmp41: ml3139, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3133, ml3139)

    # tmp17: ml3139, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3139), (finish-start)*1e-9)
end