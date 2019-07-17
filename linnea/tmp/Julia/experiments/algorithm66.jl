using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm66(ml2221::Array{Float64,2}, ml2222::Array{Float64,2}, ml2223::Array{Float64,2}, ml2224::Array{Float64,2}, ml2225::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2221, full, L: ml2222, full, A: ml2223, full, B: ml2224, full, y: ml2225, full
    ml2226 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2226, ml2224)

    # R: ml2221, full, L: ml2222, full, A: ml2223, full, y: ml2225, full, tmp26: ml2226, full
    ml2227 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2223, ml2227, info) = LinearAlgebra.LAPACK.getrf!(ml2223)

    # R: ml2221, full, L: ml2222, full, y: ml2225, full, tmp26: ml2226, full, P11: ml2227, ipiv, L9: ml2223, lower_triangular_udiag, U10: ml2223, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2223, ml2226)

    # R: ml2221, full, L: ml2222, full, y: ml2225, full, P11: ml2227, ipiv, L9: ml2223, lower_triangular_udiag, tmp27: ml2226, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2223, ml2226)

    # R: ml2221, full, L: ml2222, full, y: ml2225, full, P11: ml2227, ipiv, tmp28: ml2226, full
    ml2228 = [1:length(ml2227);]
    @inbounds for i in 1:length(ml2227)
        ml2228[i], ml2228[ml2227[i]] = ml2228[ml2227[i]], ml2228[i];
    end;
    ml2229 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2229 = ml2226[invperm(ml2228),:]

    # R: ml2221, full, L: ml2222, full, y: ml2225, full, tmp25: ml2229, full
    ml2230 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2229, 0.0, ml2230)

    # R: ml2221, full, L: ml2222, full, y: ml2225, full, tmp19: ml2230, symmetric_lower_triangular
    ml2231 = diag(ml2222)
    ml2232 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2221, 1, ml2232, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2221, 2);
        view(ml2221, :, i)[:] .*= ml2231;
    end;        

    # R: ml2232, full, y: ml2225, full, tmp19: ml2230, symmetric_lower_triangular, tmp29: ml2221, full
    for i = 1:2000-1;
        view(ml2230, i, i+1:2000)[:] = view(ml2230, i+1:2000, i);
    end;
    ml2233 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2230, 1, ml2233, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2221, ml2232, 1.0, ml2230)

    # y: ml2225, full, tmp19: ml2233, full, tmp31: ml2230, full
    ml2234 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2230, ml2234, info) = LinearAlgebra.LAPACK.getrf!(ml2230)

    # y: ml2225, full, tmp19: ml2233, full, P35: ml2234, ipiv, L33: ml2230, lower_triangular_udiag, U34: ml2230, upper_triangular
    ml2235 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2233, ml2225, 0.0, ml2235)

    # P35: ml2234, ipiv, L33: ml2230, lower_triangular_udiag, U34: ml2230, upper_triangular, tmp32: ml2235, full
    ml2236 = [1:length(ml2234);]
    @inbounds for i in 1:length(ml2234)
        ml2236[i], ml2236[ml2234[i]] = ml2236[ml2234[i]], ml2236[i];
    end;
    ml2237 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2237 = ml2235[ml2236]

    # L33: ml2230, lower_triangular_udiag, U34: ml2230, upper_triangular, tmp40: ml2237, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2230, ml2237)

    # U34: ml2230, upper_triangular, tmp41: ml2237, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2230, ml2237)

    # tmp17: ml2237, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2237), (finish-start)*1e-9)
end