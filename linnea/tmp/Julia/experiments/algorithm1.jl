using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm1(ml48::Array{Float64,2}, ml49::Array{Float64,2}, ml50::Array{Float64,2}, ml51::Array{Float64,2}, ml52::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml48, full, L: ml49, full, A: ml50, full, B: ml51, full, y: ml52, full
    ml53 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml50, ml53, info) = LinearAlgebra.LAPACK.getrf!(ml50)

    # R: ml48, full, L: ml49, full, B: ml51, full, y: ml52, full, P11: ml53, ipiv, L9: ml50, lower_triangular_udiag, U10: ml50, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml50, ml51)

    # R: ml48, full, L: ml49, full, y: ml52, full, P11: ml53, ipiv, L9: ml50, lower_triangular_udiag, tmp53: ml51, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml50, ml51)

    # R: ml48, full, L: ml49, full, y: ml52, full, P11: ml53, ipiv, tmp54: ml51, full
    ml54 = [1:length(ml53);]
    @inbounds for i in 1:length(ml53)
        ml54[i], ml54[ml53[i]] = ml54[ml53[i]], ml54[i];
    end;
    ml55 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml55 = ml51[:,invperm(ml54)]

    # R: ml48, full, L: ml49, full, y: ml52, full, tmp55: ml55, full
    ml56 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml56, ml55)

    # R: ml48, full, L: ml49, full, y: ml52, full, tmp25: ml56, full
    ml57 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml56, 0.0, ml57)

    # R: ml48, full, L: ml49, full, y: ml52, full, tmp19: ml57, symmetric_lower_triangular
    ml58 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml57, ml52, 0.0, ml58)

    # R: ml48, full, L: ml49, full, tmp19: ml57, symmetric_lower_triangular, tmp32: ml58, full
    ml59 = diag(ml49)
    ml60 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml48, 1, ml60, 1)
    # tmp29 = (L R)
    for i = 1:size(ml48, 2);
        view(ml48, :, i)[:] .*= ml59;
    end;        

    # R: ml60, full, tmp19: ml57, symmetric_lower_triangular, tmp32: ml58, full, tmp29: ml48, full
    for i = 1:2000-1;
        view(ml57, i, i+1:2000)[:] = view(ml57, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml60, ml48, 1.0, ml57)

    # tmp32: ml58, full, tmp31: ml57, full
    ml61 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml57, ml61, info) = LinearAlgebra.LAPACK.getrf!(ml57)

    # tmp32: ml58, full, P35: ml61, ipiv, L33: ml57, lower_triangular_udiag, U34: ml57, upper_triangular
    ml62 = [1:length(ml61);]
    @inbounds for i in 1:length(ml61)
        ml62[i], ml62[ml61[i]] = ml62[ml61[i]], ml62[i];
    end;
    ml63 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml63 = ml58[ml62]

    # L33: ml57, lower_triangular_udiag, U34: ml57, upper_triangular, tmp40: ml63, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml57, ml63)

    # U34: ml57, upper_triangular, tmp41: ml63, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml57, ml63)

    # tmp17: ml63, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml63), (finish-start)*1e-9)
end