using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm65(ml2187::Array{Float64,2}, ml2188::Array{Float64,2}, ml2189::Array{Float64,2}, ml2190::Array{Float64,2}, ml2191::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml2187, full, L: ml2188, full, A: ml2189, full, B: ml2190, full, y: ml2191, full
    ml2192 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2192, ml2190)

    # R: ml2187, full, L: ml2188, full, A: ml2189, full, y: ml2191, full, tmp26: ml2192, full
    ml2193 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2189, ml2193, info) = LinearAlgebra.LAPACK.getrf!(ml2189)

    # R: ml2187, full, L: ml2188, full, y: ml2191, full, tmp26: ml2192, full, P11: ml2193, ipiv, L9: ml2189, lower_triangular_udiag, U10: ml2189, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2189, ml2192)

    # R: ml2187, full, L: ml2188, full, y: ml2191, full, P11: ml2193, ipiv, L9: ml2189, lower_triangular_udiag, tmp27: ml2192, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2189, ml2192)

    # R: ml2187, full, L: ml2188, full, y: ml2191, full, P11: ml2193, ipiv, tmp28: ml2192, full
    ml2194 = [1:length(ml2193);]
    @inbounds for i in 1:length(ml2193)
        ml2194[i], ml2194[ml2193[i]] = ml2194[ml2193[i]], ml2194[i];
    end;
    ml2195 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2195 = ml2192[invperm(ml2194),:]

    # R: ml2187, full, L: ml2188, full, y: ml2191, full, tmp25: ml2195, full
    ml2196 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2195, 0.0, ml2196)

    # R: ml2187, full, L: ml2188, full, y: ml2191, full, tmp19: ml2196, symmetric_lower_triangular
    ml2197 = diag(ml2188)
    ml2198 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2187, 1, ml2198, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2187, 2);
        view(ml2187, :, i)[:] .*= ml2197;
    end;        

    # R: ml2198, full, y: ml2191, full, tmp19: ml2196, symmetric_lower_triangular, tmp29: ml2187, full
    for i = 1:2000-1;
        view(ml2196, i, i+1:2000)[:] = view(ml2196, i+1:2000, i);
    end;
    ml2199 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2196, 1, ml2199, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2187, ml2198, 1.0, ml2196)

    # y: ml2191, full, tmp19: ml2199, full, tmp31: ml2196, full
    ml2200 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2196, ml2200, info) = LinearAlgebra.LAPACK.getrf!(ml2196)

    # y: ml2191, full, tmp19: ml2199, full, P35: ml2200, ipiv, L33: ml2196, lower_triangular_udiag, U34: ml2196, upper_triangular
    ml2201 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2199, ml2191, 0.0, ml2201)

    # P35: ml2200, ipiv, L33: ml2196, lower_triangular_udiag, U34: ml2196, upper_triangular, tmp32: ml2201, full
    ml2202 = [1:length(ml2200);]
    @inbounds for i in 1:length(ml2200)
        ml2202[i], ml2202[ml2200[i]] = ml2202[ml2200[i]], ml2202[i];
    end;
    ml2203 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2203 = ml2201[ml2202]

    # L33: ml2196, lower_triangular_udiag, U34: ml2196, upper_triangular, tmp40: ml2203, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2196, ml2203)

    # U34: ml2196, upper_triangular, tmp41: ml2203, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2196, ml2203)

    # tmp17: ml2203, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2203), (finish-start)*1e-9)
end