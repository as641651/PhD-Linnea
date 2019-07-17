using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm91(ml3060::Array{Float64,2}, ml3061::Array{Float64,2}, ml3062::Array{Float64,2}, ml3063::Array{Float64,2}, ml3064::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml3060, full, L: ml3061, full, A: ml3062, full, B: ml3063, full, y: ml3064, full
    ml3065 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3065, ml3063)

    # R: ml3060, full, L: ml3061, full, A: ml3062, full, y: ml3064, full, tmp26: ml3065, full
    ml3066 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3062, ml3066, info) = LinearAlgebra.LAPACK.getrf!(ml3062)

    # R: ml3060, full, L: ml3061, full, y: ml3064, full, tmp26: ml3065, full, P11: ml3066, ipiv, L9: ml3062, lower_triangular_udiag, U10: ml3062, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3062, ml3065)

    # R: ml3060, full, L: ml3061, full, y: ml3064, full, P11: ml3066, ipiv, L9: ml3062, lower_triangular_udiag, tmp27: ml3065, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3062, ml3065)

    # R: ml3060, full, L: ml3061, full, y: ml3064, full, P11: ml3066, ipiv, tmp28: ml3065, full
    ml3067 = [1:length(ml3066);]
    @inbounds for i in 1:length(ml3066)
        ml3067[i], ml3067[ml3066[i]] = ml3067[ml3066[i]], ml3067[i];
    end;
    ml3068 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3068 = ml3065[invperm(ml3067),:]

    # R: ml3060, full, L: ml3061, full, y: ml3064, full, tmp25: ml3068, full
    ml3069 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3068, 0.0, ml3069)

    # R: ml3060, full, L: ml3061, full, y: ml3064, full, tmp19: ml3069, symmetric_lower_triangular
    ml3070 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3069, ml3064, 0.0, ml3070)

    # R: ml3060, full, L: ml3061, full, tmp19: ml3069, symmetric_lower_triangular, tmp32: ml3070, full
    ml3071 = diag(ml3061)
    ml3072 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3060, 1, ml3072, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3060, 2);
        view(ml3060, :, i)[:] .*= ml3071;
    end;        

    # R: ml3072, full, tmp19: ml3069, symmetric_lower_triangular, tmp32: ml3070, full, tmp29: ml3060, full
    for i = 1:2000-1;
        view(ml3069, i, i+1:2000)[:] = view(ml3069, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3072, ml3060, 1.0, ml3069)

    # tmp32: ml3070, full, tmp31: ml3069, full
    ml3073 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3069, ml3073, info) = LinearAlgebra.LAPACK.getrf!(ml3069)

    # tmp32: ml3070, full, P35: ml3073, ipiv, L33: ml3069, lower_triangular_udiag, U34: ml3069, upper_triangular
    ml3074 = [1:length(ml3073);]
    @inbounds for i in 1:length(ml3073)
        ml3074[i], ml3074[ml3073[i]] = ml3074[ml3073[i]], ml3074[i];
    end;
    ml3075 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3075 = ml3070[ml3074]

    # L33: ml3069, lower_triangular_udiag, U34: ml3069, upper_triangular, tmp40: ml3075, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3069, ml3075)

    # U34: ml3069, upper_triangular, tmp41: ml3075, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3069, ml3075)

    # tmp17: ml3075, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml3075), (finish-start)*1e-9)
end