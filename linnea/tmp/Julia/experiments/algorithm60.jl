using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm60(ml2024::Array{Float64,2}, ml2025::Array{Float64,2}, ml2026::Array{Float64,2}, ml2027::Array{Float64,2}, ml2028::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2024, full, L: ml2025, full, A: ml2026, full, B: ml2027, full, y: ml2028, full
    ml2029 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2029, ml2027)

    # R: ml2024, full, L: ml2025, full, A: ml2026, full, y: ml2028, full, tmp26: ml2029, full
    ml2030 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2026, ml2030, info) = LinearAlgebra.LAPACK.getrf!(ml2026)

    # R: ml2024, full, L: ml2025, full, y: ml2028, full, tmp26: ml2029, full, P11: ml2030, ipiv, L9: ml2026, lower_triangular_udiag, U10: ml2026, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2026, ml2029)

    # R: ml2024, full, L: ml2025, full, y: ml2028, full, P11: ml2030, ipiv, L9: ml2026, lower_triangular_udiag, tmp27: ml2029, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2026, ml2029)

    # R: ml2024, full, L: ml2025, full, y: ml2028, full, P11: ml2030, ipiv, tmp28: ml2029, full
    ml2031 = [1:length(ml2030);]
    @inbounds for i in 1:length(ml2030)
        ml2031[i], ml2031[ml2030[i]] = ml2031[ml2030[i]], ml2031[i];
    end;
    ml2032 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2032 = ml2029[invperm(ml2031),:]

    # R: ml2024, full, L: ml2025, full, y: ml2028, full, tmp25: ml2032, full
    ml2033 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2032, 0.0, ml2033)

    # R: ml2024, full, L: ml2025, full, y: ml2028, full, tmp19: ml2033, symmetric_lower_triangular
    ml2034 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2033, ml2028, 0.0, ml2034)

    # R: ml2024, full, L: ml2025, full, tmp19: ml2033, symmetric_lower_triangular, tmp32: ml2034, full
    ml2035 = diag(ml2025)
    ml2036 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2024, 1, ml2036, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2024, 2);
        view(ml2024, :, i)[:] .*= ml2035;
    end;        

    # R: ml2036, full, tmp19: ml2033, symmetric_lower_triangular, tmp32: ml2034, full, tmp29: ml2024, full
    for i = 1:2000-1;
        view(ml2033, i, i+1:2000)[:] = view(ml2033, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2036, ml2024, 1.0, ml2033)

    # tmp32: ml2034, full, tmp31: ml2033, full
    ml2037 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2033, ml2037, info) = LinearAlgebra.LAPACK.getrf!(ml2033)

    # tmp32: ml2034, full, P35: ml2037, ipiv, L33: ml2033, lower_triangular_udiag, U34: ml2033, upper_triangular
    ml2038 = [1:length(ml2037);]
    @inbounds for i in 1:length(ml2037)
        ml2038[i], ml2038[ml2037[i]] = ml2038[ml2037[i]], ml2038[i];
    end;
    ml2039 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2039 = ml2034[ml2038]

    # L33: ml2033, lower_triangular_udiag, U34: ml2033, upper_triangular, tmp40: ml2039, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2033, ml2039)

    # U34: ml2033, upper_triangular, tmp41: ml2039, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2033, ml2039)

    # tmp17: ml2039, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2039), (finish-start)*1e-9)
end