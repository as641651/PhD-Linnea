using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm90(ml3028::Array{Float64,2}, ml3029::Array{Float64,2}, ml3030::Array{Float64,2}, ml3031::Array{Float64,2}, ml3032::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3028, full, L: ml3029, full, A: ml3030, full, B: ml3031, full, y: ml3032, full
    ml3033 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3033, ml3031)

    # R: ml3028, full, L: ml3029, full, A: ml3030, full, y: ml3032, full, tmp26: ml3033, full
    ml3034 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3030, ml3034, info) = LinearAlgebra.LAPACK.getrf!(ml3030)

    # R: ml3028, full, L: ml3029, full, y: ml3032, full, tmp26: ml3033, full, P11: ml3034, ipiv, L9: ml3030, lower_triangular_udiag, U10: ml3030, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3030, ml3033)

    # R: ml3028, full, L: ml3029, full, y: ml3032, full, P11: ml3034, ipiv, L9: ml3030, lower_triangular_udiag, tmp27: ml3033, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3030, ml3033)

    # R: ml3028, full, L: ml3029, full, y: ml3032, full, P11: ml3034, ipiv, tmp28: ml3033, full
    ml3035 = [1:length(ml3034);]
    @inbounds for i in 1:length(ml3034)
        ml3035[i], ml3035[ml3034[i]] = ml3035[ml3034[i]], ml3035[i];
    end;
    ml3036 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3036 = ml3033[invperm(ml3035),:]

    # R: ml3028, full, L: ml3029, full, y: ml3032, full, tmp25: ml3036, full
    ml3037 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3036, 0.0, ml3037)

    # R: ml3028, full, L: ml3029, full, y: ml3032, full, tmp19: ml3037, symmetric_lower_triangular
    ml3038 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3037, ml3032, 0.0, ml3038)

    # R: ml3028, full, L: ml3029, full, tmp19: ml3037, symmetric_lower_triangular, tmp32: ml3038, full
    ml3039 = diag(ml3029)
    ml3040 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3028, 1, ml3040, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3028, 2);
        view(ml3028, :, i)[:] .*= ml3039;
    end;        

    # R: ml3040, full, tmp19: ml3037, symmetric_lower_triangular, tmp32: ml3038, full, tmp29: ml3028, full
    for i = 1:2000-1;
        view(ml3037, i, i+1:2000)[:] = view(ml3037, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3040, ml3028, 1.0, ml3037)

    # tmp32: ml3038, full, tmp31: ml3037, full
    ml3041 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3037, ml3041, info) = LinearAlgebra.LAPACK.getrf!(ml3037)

    # tmp32: ml3038, full, P35: ml3041, ipiv, L33: ml3037, lower_triangular_udiag, U34: ml3037, upper_triangular
    ml3042 = [1:length(ml3041);]
    @inbounds for i in 1:length(ml3041)
        ml3042[i], ml3042[ml3041[i]] = ml3042[ml3041[i]], ml3042[i];
    end;
    ml3043 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3043 = ml3038[ml3042]

    # L33: ml3037, lower_triangular_udiag, U34: ml3037, upper_triangular, tmp40: ml3043, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3037, ml3043)

    # U34: ml3037, upper_triangular, tmp41: ml3043, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3037, ml3043)

    # tmp17: ml3043, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3043), (finish-start)*1e-9)
end