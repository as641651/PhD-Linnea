using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm10(ml349::Array{Float64,2}, ml350::Array{Float64,2}, ml351::Array{Float64,2}, ml352::Array{Float64,2}, ml353::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml349, full, L: ml350, full, A: ml351, full, B: ml352, full, y: ml353, full
    ml354 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml351, ml354, info) = LinearAlgebra.LAPACK.getrf!(ml351)

    # R: ml349, full, L: ml350, full, B: ml352, full, y: ml353, full, P11: ml354, ipiv, L9: ml351, lower_triangular_udiag, U10: ml351, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml351, ml352)

    # R: ml349, full, L: ml350, full, y: ml353, full, P11: ml354, ipiv, L9: ml351, lower_triangular_udiag, tmp53: ml352, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml351, ml352)

    # R: ml349, full, L: ml350, full, y: ml353, full, P11: ml354, ipiv, tmp54: ml352, full
    ml355 = [1:length(ml354);]
    @inbounds for i in 1:length(ml354)
        ml355[i], ml355[ml354[i]] = ml355[ml354[i]], ml355[i];
    end;
    ml356 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml356 = ml352[:,invperm(ml355)]

    # R: ml349, full, L: ml350, full, y: ml353, full, tmp55: ml356, full
    ml357 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml357, ml356)

    # R: ml349, full, L: ml350, full, y: ml353, full, tmp25: ml357, full
    ml358 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml357, 0.0, ml358)

    # R: ml349, full, L: ml350, full, y: ml353, full, tmp19: ml358, symmetric_lower_triangular
    ml359 = diag(ml350)
    ml360 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml349, 1, ml360, 1)
    # tmp29 = (L R)
    for i = 1:size(ml349, 2);
        view(ml349, :, i)[:] .*= ml359;
    end;        

    # R: ml360, full, y: ml353, full, tmp19: ml358, symmetric_lower_triangular, tmp29: ml349, full
    for i = 1:2000-1;
        view(ml358, i, i+1:2000)[:] = view(ml358, i+1:2000, i);
    end;
    ml361 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml358, 1, ml361, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml349, ml360, 1.0, ml358)

    # y: ml353, full, tmp19: ml361, full, tmp31: ml358, full
    ml362 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml361, ml353, 0.0, ml362)

    # tmp31: ml358, full, tmp32: ml362, full
    ml363 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml358, ml363, info) = LinearAlgebra.LAPACK.getrf!(ml358)

    # tmp32: ml362, full, P35: ml363, ipiv, L33: ml358, lower_triangular_udiag, U34: ml358, upper_triangular
    ml364 = [1:length(ml363);]
    @inbounds for i in 1:length(ml363)
        ml364[i], ml364[ml363[i]] = ml364[ml363[i]], ml364[i];
    end;
    ml365 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml365 = ml362[ml364]

    # L33: ml358, lower_triangular_udiag, U34: ml358, upper_triangular, tmp40: ml365, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml358, ml365)

    # U34: ml358, upper_triangular, tmp41: ml365, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml358, ml365)

    # tmp17: ml365, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml365), (finish-start)*1e-9)
end