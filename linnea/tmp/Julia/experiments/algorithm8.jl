using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm8(ml281::Array{Float64,2}, ml282::Array{Float64,2}, ml283::Array{Float64,2}, ml284::Array{Float64,2}, ml285::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml281, full, L: ml282, full, A: ml283, full, B: ml284, full, y: ml285, full
    ml286 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml283, ml286, info) = LinearAlgebra.LAPACK.getrf!(ml283)

    # R: ml281, full, L: ml282, full, B: ml284, full, y: ml285, full, P11: ml286, ipiv, L9: ml283, lower_triangular_udiag, U10: ml283, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml283, ml284)

    # R: ml281, full, L: ml282, full, y: ml285, full, P11: ml286, ipiv, L9: ml283, lower_triangular_udiag, tmp53: ml284, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml283, ml284)

    # R: ml281, full, L: ml282, full, y: ml285, full, P11: ml286, ipiv, tmp54: ml284, full
    ml287 = [1:length(ml286);]
    @inbounds for i in 1:length(ml286)
        ml287[i], ml287[ml286[i]] = ml287[ml286[i]], ml287[i];
    end;
    ml288 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml288 = ml284[:,invperm(ml287)]

    # R: ml281, full, L: ml282, full, y: ml285, full, tmp55: ml288, full
    ml289 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml289, ml288)

    # R: ml281, full, L: ml282, full, y: ml285, full, tmp25: ml289, full
    ml290 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml289, 0.0, ml290)

    # R: ml281, full, L: ml282, full, y: ml285, full, tmp19: ml290, symmetric_lower_triangular
    ml291 = diag(ml282)
    ml292 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml281, 1, ml292, 1)
    # tmp29 = (L R)
    for i = 1:size(ml281, 2);
        view(ml281, :, i)[:] .*= ml291;
    end;        

    # R: ml292, full, y: ml285, full, tmp19: ml290, symmetric_lower_triangular, tmp29: ml281, full
    for i = 1:2000-1;
        view(ml290, i, i+1:2000)[:] = view(ml290, i+1:2000, i);
    end;
    ml293 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml290, 1, ml293, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml281, ml292, 1.0, ml290)

    # y: ml285, full, tmp19: ml293, full, tmp31: ml290, full
    ml294 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml290, ml294, info) = LinearAlgebra.LAPACK.getrf!(ml290)

    # y: ml285, full, tmp19: ml293, full, P35: ml294, ipiv, L33: ml290, lower_triangular_udiag, U34: ml290, upper_triangular
    ml295 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml293, ml285, 0.0, ml295)

    # P35: ml294, ipiv, L33: ml290, lower_triangular_udiag, U34: ml290, upper_triangular, tmp32: ml295, full
    ml296 = [1:length(ml294);]
    @inbounds for i in 1:length(ml294)
        ml296[i], ml296[ml294[i]] = ml296[ml294[i]], ml296[i];
    end;
    ml297 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml297 = ml295[ml296]

    # L33: ml290, lower_triangular_udiag, U34: ml290, upper_triangular, tmp40: ml297, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml290, ml297)

    # U34: ml290, upper_triangular, tmp41: ml297, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml290, ml297)

    # tmp17: ml297, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml297), (finish-start)*1e-9)
end