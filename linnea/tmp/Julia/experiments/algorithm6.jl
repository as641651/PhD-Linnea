using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm6(ml213::Array{Float64,2}, ml214::Array{Float64,2}, ml215::Array{Float64,2}, ml216::Array{Float64,2}, ml217::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml213, full, L: ml214, full, A: ml215, full, B: ml216, full, y: ml217, full
    ml218 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml215, ml218, info) = LinearAlgebra.LAPACK.getrf!(ml215)

    # R: ml213, full, L: ml214, full, B: ml216, full, y: ml217, full, P11: ml218, ipiv, L9: ml215, lower_triangular_udiag, U10: ml215, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml215, ml216)

    # R: ml213, full, L: ml214, full, y: ml217, full, P11: ml218, ipiv, L9: ml215, lower_triangular_udiag, tmp53: ml216, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml215, ml216)

    # R: ml213, full, L: ml214, full, y: ml217, full, P11: ml218, ipiv, tmp54: ml216, full
    ml219 = [1:length(ml218);]
    @inbounds for i in 1:length(ml218)
        ml219[i], ml219[ml218[i]] = ml219[ml218[i]], ml219[i];
    end;
    ml220 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml220 = ml216[:,invperm(ml219)]

    # R: ml213, full, L: ml214, full, y: ml217, full, tmp55: ml220, full
    ml221 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml221, ml220)

    # R: ml213, full, L: ml214, full, y: ml217, full, tmp25: ml221, full
    ml222 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml221, 0.0, ml222)

    # R: ml213, full, L: ml214, full, y: ml217, full, tmp19: ml222, symmetric_lower_triangular
    ml223 = diag(ml214)
    ml224 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml213, 1, ml224, 1)
    # tmp29 = (L R)
    for i = 1:size(ml213, 2);
        view(ml213, :, i)[:] .*= ml223;
    end;        

    # R: ml224, full, y: ml217, full, tmp19: ml222, symmetric_lower_triangular, tmp29: ml213, full
    for i = 1:2000-1;
        view(ml222, i, i+1:2000)[:] = view(ml222, i+1:2000, i);
    end;
    ml225 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml222, 1, ml225, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml213, ml224, 1.0, ml222)

    # y: ml217, full, tmp19: ml225, full, tmp31: ml222, full
    ml226 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml222, ml226, info) = LinearAlgebra.LAPACK.getrf!(ml222)

    # y: ml217, full, tmp19: ml225, full, P35: ml226, ipiv, L33: ml222, lower_triangular_udiag, U34: ml222, upper_triangular
    ml227 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml225, ml217, 0.0, ml227)

    # P35: ml226, ipiv, L33: ml222, lower_triangular_udiag, U34: ml222, upper_triangular, tmp32: ml227, full
    ml228 = [1:length(ml226);]
    @inbounds for i in 1:length(ml226)
        ml228[i], ml228[ml226[i]] = ml228[ml226[i]], ml228[i];
    end;
    ml229 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml229 = ml227[ml228]

    # L33: ml222, lower_triangular_udiag, U34: ml222, upper_triangular, tmp40: ml229, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml222, ml229)

    # U34: ml222, upper_triangular, tmp41: ml229, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml222, ml229)

    # tmp17: ml229, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml229), (finish-start)*1e-9)
end