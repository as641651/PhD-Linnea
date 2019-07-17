using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm30(ml1020::Array{Float64,2}, ml1021::Array{Float64,2}, ml1022::Array{Float64,2}, ml1023::Array{Float64,2}, ml1024::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1020, full, L: ml1021, full, A: ml1022, full, B: ml1023, full, y: ml1024, full
    ml1025 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1025, ml1023)

    # R: ml1020, full, L: ml1021, full, A: ml1022, full, y: ml1024, full, tmp26: ml1025, full
    ml1026 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1022, ml1026, info) = LinearAlgebra.LAPACK.getrf!(ml1022)

    # R: ml1020, full, L: ml1021, full, y: ml1024, full, tmp26: ml1025, full, P11: ml1026, ipiv, L9: ml1022, lower_triangular_udiag, U10: ml1022, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1022, ml1025)

    # R: ml1020, full, L: ml1021, full, y: ml1024, full, P11: ml1026, ipiv, L9: ml1022, lower_triangular_udiag, tmp27: ml1025, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1022, ml1025)

    # R: ml1020, full, L: ml1021, full, y: ml1024, full, P11: ml1026, ipiv, tmp28: ml1025, full
    ml1027 = [1:length(ml1026);]
    @inbounds for i in 1:length(ml1026)
        ml1027[i], ml1027[ml1026[i]] = ml1027[ml1026[i]], ml1027[i];
    end;
    ml1028 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1028 = ml1025[invperm(ml1027),:]

    # R: ml1020, full, L: ml1021, full, y: ml1024, full, tmp25: ml1028, full
    ml1029 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1028, 0.0, ml1029)

    # R: ml1020, full, L: ml1021, full, y: ml1024, full, tmp19: ml1029, symmetric_lower_triangular
    ml1030 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1029, ml1024, 0.0, ml1030)

    # R: ml1020, full, L: ml1021, full, tmp19: ml1029, symmetric_lower_triangular, tmp32: ml1030, full
    ml1031 = diag(ml1021)
    ml1032 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1020, 1, ml1032, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1020, 2);
        view(ml1020, :, i)[:] .*= ml1031;
    end;        

    # R: ml1032, full, tmp19: ml1029, symmetric_lower_triangular, tmp32: ml1030, full, tmp29: ml1020, full
    for i = 1:2000-1;
        view(ml1029, i, i+1:2000)[:] = view(ml1029, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1032, ml1020, 1.0, ml1029)

    # tmp32: ml1030, full, tmp31: ml1029, full
    ml1033 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1029, ml1033, info) = LinearAlgebra.LAPACK.getrf!(ml1029)

    # tmp32: ml1030, full, P35: ml1033, ipiv, L33: ml1029, lower_triangular_udiag, U34: ml1029, upper_triangular
    ml1034 = [1:length(ml1033);]
    @inbounds for i in 1:length(ml1033)
        ml1034[i], ml1034[ml1033[i]] = ml1034[ml1033[i]], ml1034[i];
    end;
    ml1035 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1035 = ml1030[ml1034]

    # L33: ml1029, lower_triangular_udiag, U34: ml1029, upper_triangular, tmp40: ml1035, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1029, ml1035)

    # U34: ml1029, upper_triangular, tmp41: ml1035, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1029, ml1035)

    # tmp17: ml1035, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1035), (finish-start)*1e-9)
end