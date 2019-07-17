using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm10(ml332::Array{Float64,2}, ml333::Array{Float64,2}, ml334::Array{Float64,2}, ml335::Array{Float64,2}, ml336::Array{Float64,1})
    # cost 5.07e+10
    # R: ml332, full, L: ml333, full, A: ml334, full, B: ml335, full, y: ml336, full
    ml337 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml334, ml337, info) = LinearAlgebra.LAPACK.getrf!(ml334)

    # R: ml332, full, L: ml333, full, B: ml335, full, y: ml336, full, P11: ml337, ipiv, L9: ml334, lower_triangular_udiag, U10: ml334, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml334, ml335)

    # R: ml332, full, L: ml333, full, y: ml336, full, P11: ml337, ipiv, L9: ml334, lower_triangular_udiag, tmp53: ml335, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml334, ml335)

    # R: ml332, full, L: ml333, full, y: ml336, full, P11: ml337, ipiv, tmp54: ml335, full
    ml338 = [1:length(ml337);]
    @inbounds for i in 1:length(ml337)
        ml338[i], ml338[ml337[i]] = ml338[ml337[i]], ml338[i];
    end;
    ml339 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml339 = ml335[:,invperm(ml338)]

    # R: ml332, full, L: ml333, full, y: ml336, full, tmp55: ml339, full
    ml340 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml340, ml339)

    # R: ml332, full, L: ml333, full, y: ml336, full, tmp25: ml340, full
    ml341 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml340, 0.0, ml341)

    # R: ml332, full, L: ml333, full, y: ml336, full, tmp19: ml341, symmetric_lower_triangular
    ml342 = diag(ml333)
    ml343 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml332, 1, ml343, 1)
    # tmp29 = (L R)
    for i = 1:size(ml332, 2);
        view(ml332, :, i)[:] .*= ml342;
    end;        

    # R: ml343, full, y: ml336, full, tmp19: ml341, symmetric_lower_triangular, tmp29: ml332, full
    for i = 1:2000-1;
        view(ml341, i, i+1:2000)[:] = view(ml341, i+1:2000, i);
    end;
    ml344 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml341, 1, ml344, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml332, ml343, 1.0, ml341)

    # y: ml336, full, tmp19: ml344, full, tmp31: ml341, full
    ml345 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml344, ml336, 0.0, ml345)

    # tmp31: ml341, full, tmp32: ml345, full
    ml346 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml341, ml346, info) = LinearAlgebra.LAPACK.getrf!(ml341)

    # tmp32: ml345, full, P35: ml346, ipiv, L33: ml341, lower_triangular_udiag, U34: ml341, upper_triangular
    ml347 = [1:length(ml346);]
    @inbounds for i in 1:length(ml346)
        ml347[i], ml347[ml346[i]] = ml347[ml346[i]], ml347[i];
    end;
    ml348 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml348 = ml345[ml347]

    # L33: ml341, lower_triangular_udiag, U34: ml341, upper_triangular, tmp40: ml348, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml341, ml348)

    # U34: ml341, upper_triangular, tmp41: ml348, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml341, ml348)

    # tmp17: ml348, full
    # x = tmp17
    return (ml348)
end