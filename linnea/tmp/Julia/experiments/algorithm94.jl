using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm94(ml3157::Array{Float64,2}, ml3158::Array{Float64,2}, ml3159::Array{Float64,2}, ml3160::Array{Float64,2}, ml3161::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3157, full, L: ml3158, full, A: ml3159, full, B: ml3160, full, y: ml3161, full
    ml3162 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3162, ml3160)

    # R: ml3157, full, L: ml3158, full, A: ml3159, full, y: ml3161, full, tmp26: ml3162, full
    ml3163 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3159, ml3163, info) = LinearAlgebra.LAPACK.getrf!(ml3159)

    # R: ml3157, full, L: ml3158, full, y: ml3161, full, tmp26: ml3162, full, P11: ml3163, ipiv, L9: ml3159, lower_triangular_udiag, U10: ml3159, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3159, ml3162)

    # R: ml3157, full, L: ml3158, full, y: ml3161, full, P11: ml3163, ipiv, L9: ml3159, lower_triangular_udiag, tmp27: ml3162, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3159, ml3162)

    # R: ml3157, full, L: ml3158, full, y: ml3161, full, P11: ml3163, ipiv, tmp28: ml3162, full
    ml3164 = [1:length(ml3163);]
    @inbounds for i in 1:length(ml3163)
        ml3164[i], ml3164[ml3163[i]] = ml3164[ml3163[i]], ml3164[i];
    end;
    ml3165 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3165 = ml3162[invperm(ml3164),:]

    # R: ml3157, full, L: ml3158, full, y: ml3161, full, tmp25: ml3165, full
    ml3166 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3165, 0.0, ml3166)

    # R: ml3157, full, L: ml3158, full, y: ml3161, full, tmp19: ml3166, symmetric_lower_triangular
    ml3167 = diag(ml3158)
    ml3168 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3157, 1, ml3168, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3157, 2);
        view(ml3157, :, i)[:] .*= ml3167;
    end;        

    # R: ml3168, full, y: ml3161, full, tmp19: ml3166, symmetric_lower_triangular, tmp29: ml3157, full
    for i = 1:2000-1;
        view(ml3166, i, i+1:2000)[:] = view(ml3166, i+1:2000, i);
    end;
    ml3169 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3166, 1, ml3169, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3157, ml3168, 1.0, ml3166)

    # y: ml3161, full, tmp19: ml3169, full, tmp31: ml3166, full
    ml3170 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3166, ml3170, info) = LinearAlgebra.LAPACK.getrf!(ml3166)

    # y: ml3161, full, tmp19: ml3169, full, P35: ml3170, ipiv, L33: ml3166, lower_triangular_udiag, U34: ml3166, upper_triangular
    ml3171 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3169, ml3161, 0.0, ml3171)

    # P35: ml3170, ipiv, L33: ml3166, lower_triangular_udiag, U34: ml3166, upper_triangular, tmp32: ml3171, full
    ml3172 = [1:length(ml3170);]
    @inbounds for i in 1:length(ml3170)
        ml3172[i], ml3172[ml3170[i]] = ml3172[ml3170[i]], ml3172[i];
    end;
    ml3173 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3173 = ml3171[ml3172]

    # L33: ml3166, lower_triangular_udiag, U34: ml3166, upper_triangular, tmp40: ml3173, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3166, ml3173)

    # U34: ml3166, upper_triangular, tmp41: ml3173, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3166, ml3173)

    # tmp17: ml3173, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3173), (finish-start)*1e-9)
end