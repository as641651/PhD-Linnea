using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm7(ml247::Array{Float64,2}, ml248::Array{Float64,2}, ml249::Array{Float64,2}, ml250::Array{Float64,2}, ml251::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml247, full, L: ml248, full, A: ml249, full, B: ml250, full, y: ml251, full
    ml252 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml249, ml252, info) = LinearAlgebra.LAPACK.getrf!(ml249)

    # R: ml247, full, L: ml248, full, B: ml250, full, y: ml251, full, P11: ml252, ipiv, L9: ml249, lower_triangular_udiag, U10: ml249, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml249, ml250)

    # R: ml247, full, L: ml248, full, y: ml251, full, P11: ml252, ipiv, L9: ml249, lower_triangular_udiag, tmp53: ml250, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml249, ml250)

    # R: ml247, full, L: ml248, full, y: ml251, full, P11: ml252, ipiv, tmp54: ml250, full
    ml253 = [1:length(ml252);]
    @inbounds for i in 1:length(ml252)
        ml253[i], ml253[ml252[i]] = ml253[ml252[i]], ml253[i];
    end;
    ml254 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml254 = ml250[:,invperm(ml253)]

    # R: ml247, full, L: ml248, full, y: ml251, full, tmp55: ml254, full
    ml255 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml255, ml254)

    # R: ml247, full, L: ml248, full, y: ml251, full, tmp25: ml255, full
    ml256 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml255, 0.0, ml256)

    # R: ml247, full, L: ml248, full, y: ml251, full, tmp19: ml256, symmetric_lower_triangular
    ml257 = diag(ml248)
    ml258 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml247, 1, ml258, 1)
    # tmp29 = (L R)
    for i = 1:size(ml247, 2);
        view(ml247, :, i)[:] .*= ml257;
    end;        

    # R: ml258, full, y: ml251, full, tmp19: ml256, symmetric_lower_triangular, tmp29: ml247, full
    for i = 1:2000-1;
        view(ml256, i, i+1:2000)[:] = view(ml256, i+1:2000, i);
    end;
    ml259 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml256, 1, ml259, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml247, ml258, 1.0, ml256)

    # y: ml251, full, tmp19: ml259, full, tmp31: ml256, full
    ml260 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml256, ml260, info) = LinearAlgebra.LAPACK.getrf!(ml256)

    # y: ml251, full, tmp19: ml259, full, P35: ml260, ipiv, L33: ml256, lower_triangular_udiag, U34: ml256, upper_triangular
    ml261 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml259, ml251, 0.0, ml261)

    # P35: ml260, ipiv, L33: ml256, lower_triangular_udiag, U34: ml256, upper_triangular, tmp32: ml261, full
    ml262 = [1:length(ml260);]
    @inbounds for i in 1:length(ml260)
        ml262[i], ml262[ml260[i]] = ml262[ml260[i]], ml262[i];
    end;
    ml263 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml263 = ml261[ml262]

    # L33: ml256, lower_triangular_udiag, U34: ml256, upper_triangular, tmp40: ml263, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml256, ml263)

    # U34: ml256, upper_triangular, tmp41: ml263, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml256, ml263)

    # tmp17: ml263, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml263), (finish-start)*1e-9)
end