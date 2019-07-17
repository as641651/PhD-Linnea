using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm3(ml112::Array{Float64,2}, ml113::Array{Float64,2}, ml114::Array{Float64,2}, ml115::Array{Float64,2}, ml116::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml112, full, L: ml113, full, A: ml114, full, B: ml115, full, y: ml116, full
    ml117 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml114, ml117, info) = LinearAlgebra.LAPACK.getrf!(ml114)

    # R: ml112, full, L: ml113, full, B: ml115, full, y: ml116, full, P11: ml117, ipiv, L9: ml114, lower_triangular_udiag, U10: ml114, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml114, ml115)

    # R: ml112, full, L: ml113, full, y: ml116, full, P11: ml117, ipiv, L9: ml114, lower_triangular_udiag, tmp53: ml115, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml114, ml115)

    # R: ml112, full, L: ml113, full, y: ml116, full, P11: ml117, ipiv, tmp54: ml115, full
    ml118 = [1:length(ml117);]
    @inbounds for i in 1:length(ml117)
        ml118[i], ml118[ml117[i]] = ml118[ml117[i]], ml118[i];
    end;
    ml119 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml119 = ml115[:,invperm(ml118)]

    # R: ml112, full, L: ml113, full, y: ml116, full, tmp55: ml119, full
    ml120 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml120, ml119)

    # R: ml112, full, L: ml113, full, y: ml116, full, tmp25: ml120, full
    ml121 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml120, 0.0, ml121)

    # R: ml112, full, L: ml113, full, y: ml116, full, tmp19: ml121, symmetric_lower_triangular
    ml122 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml121, ml116, 0.0, ml122)

    # R: ml112, full, L: ml113, full, tmp19: ml121, symmetric_lower_triangular, tmp32: ml122, full
    ml123 = diag(ml113)
    ml124 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml112, 1, ml124, 1)
    # tmp29 = (L R)
    for i = 1:size(ml112, 2);
        view(ml112, :, i)[:] .*= ml123;
    end;        

    # R: ml124, full, tmp19: ml121, symmetric_lower_triangular, tmp32: ml122, full, tmp29: ml112, full
    for i = 1:2000-1;
        view(ml121, i, i+1:2000)[:] = view(ml121, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml124, ml112, 1.0, ml121)

    # tmp32: ml122, full, tmp31: ml121, full
    ml125 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml121, ml125, info) = LinearAlgebra.LAPACK.getrf!(ml121)

    # tmp32: ml122, full, P35: ml125, ipiv, L33: ml121, lower_triangular_udiag, U34: ml121, upper_triangular
    ml126 = [1:length(ml125);]
    @inbounds for i in 1:length(ml125)
        ml126[i], ml126[ml125[i]] = ml126[ml125[i]], ml126[i];
    end;
    ml127 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml127 = ml122[ml126]

    # L33: ml121, lower_triangular_udiag, U34: ml121, upper_triangular, tmp40: ml127, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml121, ml127)

    # U34: ml121, upper_triangular, tmp41: ml127, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml121, ml127)

    # tmp17: ml127, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml127), (finish-start)*1e-9)
end