using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm68(ml2289::Array{Float64,2}, ml2290::Array{Float64,2}, ml2291::Array{Float64,2}, ml2292::Array{Float64,2}, ml2293::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2289, full, L: ml2290, full, A: ml2291, full, B: ml2292, full, y: ml2293, full
    ml2294 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2294, ml2292)

    # R: ml2289, full, L: ml2290, full, A: ml2291, full, y: ml2293, full, tmp26: ml2294, full
    ml2295 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2291, ml2295, info) = LinearAlgebra.LAPACK.getrf!(ml2291)

    # R: ml2289, full, L: ml2290, full, y: ml2293, full, tmp26: ml2294, full, P11: ml2295, ipiv, L9: ml2291, lower_triangular_udiag, U10: ml2291, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2291, ml2294)

    # R: ml2289, full, L: ml2290, full, y: ml2293, full, P11: ml2295, ipiv, L9: ml2291, lower_triangular_udiag, tmp27: ml2294, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2291, ml2294)

    # R: ml2289, full, L: ml2290, full, y: ml2293, full, P11: ml2295, ipiv, tmp28: ml2294, full
    ml2296 = [1:length(ml2295);]
    @inbounds for i in 1:length(ml2295)
        ml2296[i], ml2296[ml2295[i]] = ml2296[ml2295[i]], ml2296[i];
    end;
    ml2297 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2297 = ml2294[invperm(ml2296),:]

    # R: ml2289, full, L: ml2290, full, y: ml2293, full, tmp25: ml2297, full
    ml2298 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2297, 0.0, ml2298)

    # R: ml2289, full, L: ml2290, full, y: ml2293, full, tmp19: ml2298, symmetric_lower_triangular
    ml2299 = diag(ml2290)
    ml2300 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2289, 1, ml2300, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2289, 2);
        view(ml2289, :, i)[:] .*= ml2299;
    end;        

    # R: ml2300, full, y: ml2293, full, tmp19: ml2298, symmetric_lower_triangular, tmp29: ml2289, full
    for i = 1:2000-1;
        view(ml2298, i, i+1:2000)[:] = view(ml2298, i+1:2000, i);
    end;
    ml2301 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2298, 1, ml2301, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2289, ml2300, 1.0, ml2298)

    # y: ml2293, full, tmp19: ml2301, full, tmp31: ml2298, full
    ml2302 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2298, ml2302, info) = LinearAlgebra.LAPACK.getrf!(ml2298)

    # y: ml2293, full, tmp19: ml2301, full, P35: ml2302, ipiv, L33: ml2298, lower_triangular_udiag, U34: ml2298, upper_triangular
    ml2303 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2301, ml2293, 0.0, ml2303)

    # P35: ml2302, ipiv, L33: ml2298, lower_triangular_udiag, U34: ml2298, upper_triangular, tmp32: ml2303, full
    ml2304 = [1:length(ml2302);]
    @inbounds for i in 1:length(ml2302)
        ml2304[i], ml2304[ml2302[i]] = ml2304[ml2302[i]], ml2304[i];
    end;
    ml2305 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2305 = ml2303[ml2304]

    # L33: ml2298, lower_triangular_udiag, U34: ml2298, upper_triangular, tmp40: ml2305, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2298, ml2305)

    # U34: ml2298, upper_triangular, tmp41: ml2305, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2298, ml2305)

    # tmp17: ml2305, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2305), (finish-start)*1e-9)
end