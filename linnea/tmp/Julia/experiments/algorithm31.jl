using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm31(ml1052::Array{Float64,2}, ml1053::Array{Float64,2}, ml1054::Array{Float64,2}, ml1055::Array{Float64,2}, ml1056::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1052, full, L: ml1053, full, A: ml1054, full, B: ml1055, full, y: ml1056, full
    ml1057 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1057, ml1055)

    # R: ml1052, full, L: ml1053, full, A: ml1054, full, y: ml1056, full, tmp26: ml1057, full
    ml1058 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1054, ml1058, info) = LinearAlgebra.LAPACK.getrf!(ml1054)

    # R: ml1052, full, L: ml1053, full, y: ml1056, full, tmp26: ml1057, full, P11: ml1058, ipiv, L9: ml1054, lower_triangular_udiag, U10: ml1054, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1054, ml1057)

    # R: ml1052, full, L: ml1053, full, y: ml1056, full, P11: ml1058, ipiv, L9: ml1054, lower_triangular_udiag, tmp27: ml1057, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1054, ml1057)

    # R: ml1052, full, L: ml1053, full, y: ml1056, full, P11: ml1058, ipiv, tmp28: ml1057, full
    ml1059 = [1:length(ml1058);]
    @inbounds for i in 1:length(ml1058)
        ml1059[i], ml1059[ml1058[i]] = ml1059[ml1058[i]], ml1059[i];
    end;
    ml1060 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1060 = ml1057[invperm(ml1059),:]

    # R: ml1052, full, L: ml1053, full, y: ml1056, full, tmp25: ml1060, full
    ml1061 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1060, 0.0, ml1061)

    # R: ml1052, full, L: ml1053, full, y: ml1056, full, tmp19: ml1061, symmetric_lower_triangular
    ml1062 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1061, ml1056, 0.0, ml1062)

    # R: ml1052, full, L: ml1053, full, tmp19: ml1061, symmetric_lower_triangular, tmp32: ml1062, full
    ml1063 = diag(ml1053)
    ml1064 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1052, 1, ml1064, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1052, 2);
        view(ml1052, :, i)[:] .*= ml1063;
    end;        

    # R: ml1064, full, tmp19: ml1061, symmetric_lower_triangular, tmp32: ml1062, full, tmp29: ml1052, full
    for i = 1:2000-1;
        view(ml1061, i, i+1:2000)[:] = view(ml1061, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1064, ml1052, 1.0, ml1061)

    # tmp32: ml1062, full, tmp31: ml1061, full
    ml1065 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1061, ml1065, info) = LinearAlgebra.LAPACK.getrf!(ml1061)

    # tmp32: ml1062, full, P35: ml1065, ipiv, L33: ml1061, lower_triangular_udiag, U34: ml1061, upper_triangular
    ml1066 = [1:length(ml1065);]
    @inbounds for i in 1:length(ml1065)
        ml1066[i], ml1066[ml1065[i]] = ml1066[ml1065[i]], ml1066[i];
    end;
    ml1067 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1067 = ml1062[ml1066]

    # L33: ml1061, lower_triangular_udiag, U34: ml1061, upper_triangular, tmp40: ml1067, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1061, ml1067)

    # U34: ml1061, upper_triangular, tmp41: ml1067, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1061, ml1067)

    # tmp17: ml1067, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1067), (finish-start)*1e-9)
end