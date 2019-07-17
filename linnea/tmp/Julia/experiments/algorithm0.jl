using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm0(ml16::Array{Float64,2}, ml17::Array{Float64,2}, ml18::Array{Float64,2}, ml19::Array{Float64,2}, ml20::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml16, full, L: ml17, full, A: ml18, full, B: ml19, full, y: ml20, full
    ml21 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml18, ml21, info) = LinearAlgebra.LAPACK.getrf!(ml18)

    # R: ml16, full, L: ml17, full, B: ml19, full, y: ml20, full, P11: ml21, ipiv, L9: ml18, lower_triangular_udiag, U10: ml18, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml18, ml19)

    # R: ml16, full, L: ml17, full, y: ml20, full, P11: ml21, ipiv, L9: ml18, lower_triangular_udiag, tmp53: ml19, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml18, ml19)

    # R: ml16, full, L: ml17, full, y: ml20, full, P11: ml21, ipiv, tmp54: ml19, full
    ml22 = [1:length(ml21);]
    @inbounds for i in 1:length(ml21)
        ml22[i], ml22[ml21[i]] = ml22[ml21[i]], ml22[i];
    end;
    ml23 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml23 = ml19[:,invperm(ml22)]

    # R: ml16, full, L: ml17, full, y: ml20, full, tmp55: ml23, full
    ml24 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml24, ml23)

    # R: ml16, full, L: ml17, full, y: ml20, full, tmp25: ml24, full
    ml25 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml24, 0.0, ml25)

    # R: ml16, full, L: ml17, full, y: ml20, full, tmp19: ml25, symmetric_lower_triangular
    ml26 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml25, ml20, 0.0, ml26)

    # R: ml16, full, L: ml17, full, tmp19: ml25, symmetric_lower_triangular, tmp32: ml26, full
    ml27 = diag(ml17)
    ml28 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml16, 1, ml28, 1)
    # tmp29 = (L R)
    for i = 1:size(ml16, 2);
        view(ml16, :, i)[:] .*= ml27;
    end;        

    # R: ml28, full, tmp19: ml25, symmetric_lower_triangular, tmp32: ml26, full, tmp29: ml16, full
    for i = 1:2000-1;
        view(ml25, i, i+1:2000)[:] = view(ml25, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml28, ml16, 1.0, ml25)

    # tmp32: ml26, full, tmp31: ml25, full
    ml29 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml25, ml29, info) = LinearAlgebra.LAPACK.getrf!(ml25)

    # tmp32: ml26, full, P35: ml29, ipiv, L33: ml25, lower_triangular_udiag, U34: ml25, upper_triangular
    ml30 = [1:length(ml29);]
    @inbounds for i in 1:length(ml29)
        ml30[i], ml30[ml29[i]] = ml30[ml29[i]], ml30[i];
    end;
    ml31 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml31 = ml26[ml30]

    # L33: ml25, lower_triangular_udiag, U34: ml25, upper_triangular, tmp40: ml31, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml25, ml31)

    # U34: ml25, upper_triangular, tmp41: ml31, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml25, ml31)

    # tmp17: ml31, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml31), (finish-start)*1e-9)
end