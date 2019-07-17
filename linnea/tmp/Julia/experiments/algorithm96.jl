using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm96(ml3225::Array{Float64,2}, ml3226::Array{Float64,2}, ml3227::Array{Float64,2}, ml3228::Array{Float64,2}, ml3229::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3225, full, L: ml3226, full, A: ml3227, full, B: ml3228, full, y: ml3229, full
    ml3230 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3230, ml3228)

    # R: ml3225, full, L: ml3226, full, A: ml3227, full, y: ml3229, full, tmp26: ml3230, full
    ml3231 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3227, ml3231, info) = LinearAlgebra.LAPACK.getrf!(ml3227)

    # R: ml3225, full, L: ml3226, full, y: ml3229, full, tmp26: ml3230, full, P11: ml3231, ipiv, L9: ml3227, lower_triangular_udiag, U10: ml3227, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3227, ml3230)

    # R: ml3225, full, L: ml3226, full, y: ml3229, full, P11: ml3231, ipiv, L9: ml3227, lower_triangular_udiag, tmp27: ml3230, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3227, ml3230)

    # R: ml3225, full, L: ml3226, full, y: ml3229, full, P11: ml3231, ipiv, tmp28: ml3230, full
    ml3232 = [1:length(ml3231);]
    @inbounds for i in 1:length(ml3231)
        ml3232[i], ml3232[ml3231[i]] = ml3232[ml3231[i]], ml3232[i];
    end;
    ml3233 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3233 = ml3230[invperm(ml3232),:]

    # R: ml3225, full, L: ml3226, full, y: ml3229, full, tmp25: ml3233, full
    ml3234 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3233, 0.0, ml3234)

    # R: ml3225, full, L: ml3226, full, y: ml3229, full, tmp19: ml3234, symmetric_lower_triangular
    ml3235 = diag(ml3226)
    ml3236 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3225, 1, ml3236, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3225, 2);
        view(ml3225, :, i)[:] .*= ml3235;
    end;        

    # R: ml3236, full, y: ml3229, full, tmp19: ml3234, symmetric_lower_triangular, tmp29: ml3225, full
    for i = 1:2000-1;
        view(ml3234, i, i+1:2000)[:] = view(ml3234, i+1:2000, i);
    end;
    ml3237 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3234, 1, ml3237, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3225, ml3236, 1.0, ml3234)

    # y: ml3229, full, tmp19: ml3237, full, tmp31: ml3234, full
    ml3238 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3234, ml3238, info) = LinearAlgebra.LAPACK.getrf!(ml3234)

    # y: ml3229, full, tmp19: ml3237, full, P35: ml3238, ipiv, L33: ml3234, lower_triangular_udiag, U34: ml3234, upper_triangular
    ml3239 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3237, ml3229, 0.0, ml3239)

    # P35: ml3238, ipiv, L33: ml3234, lower_triangular_udiag, U34: ml3234, upper_triangular, tmp32: ml3239, full
    ml3240 = [1:length(ml3238);]
    @inbounds for i in 1:length(ml3238)
        ml3240[i], ml3240[ml3238[i]] = ml3240[ml3238[i]], ml3240[i];
    end;
    ml3241 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3241 = ml3239[ml3240]

    # L33: ml3234, lower_triangular_udiag, U34: ml3234, upper_triangular, tmp40: ml3241, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3234, ml3241)

    # U34: ml3234, upper_triangular, tmp41: ml3241, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3234, ml3241)

    # tmp17: ml3241, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3241), (finish-start)*1e-9)
end