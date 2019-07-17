using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm21(ml720::Array{Float64,2}, ml721::Array{Float64,2}, ml722::Array{Float64,2}, ml723::Array{Float64,2}, ml724::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml720, full, L: ml721, full, A: ml722, full, B: ml723, full, y: ml724, full
    ml725 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml722, ml725, info) = LinearAlgebra.LAPACK.getrf!(ml722)

    # R: ml720, full, L: ml721, full, B: ml723, full, y: ml724, full, P11: ml725, ipiv, L9: ml722, lower_triangular_udiag, U10: ml722, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml722, ml723)

    # R: ml720, full, L: ml721, full, y: ml724, full, P11: ml725, ipiv, L9: ml722, lower_triangular_udiag, tmp53: ml723, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml722, ml723)

    # R: ml720, full, L: ml721, full, y: ml724, full, P11: ml725, ipiv, tmp54: ml723, full
    ml726 = [1:length(ml725);]
    @inbounds for i in 1:length(ml725)
        ml726[i], ml726[ml725[i]] = ml726[ml725[i]], ml726[i];
    end;
    ml727 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml727 = ml723[:,invperm(ml726)]

    # R: ml720, full, L: ml721, full, y: ml724, full, tmp55: ml727, full
    ml728 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml728, ml727)

    # R: ml720, full, L: ml721, full, y: ml724, full, tmp25: ml728, full
    ml729 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml728, 0.0, ml729)

    # R: ml720, full, L: ml721, full, y: ml724, full, tmp19: ml729, symmetric_lower_triangular
    ml730 = diag(ml721)
    ml731 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml720, 1, ml731, 1)
    # tmp29 = (L R)
    for i = 1:size(ml720, 2);
        view(ml720, :, i)[:] .*= ml730;
    end;        

    # R: ml731, full, y: ml724, full, tmp19: ml729, symmetric_lower_triangular, tmp29: ml720, full
    ml732 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml729, ml724, 0.0, ml732)

    # R: ml731, full, tmp19: ml729, symmetric_lower_triangular, tmp29: ml720, full, tmp32: ml732, full
    for i = 1:2000-1;
        view(ml729, i, i+1:2000)[:] = view(ml729, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml720, ml731, 1.0, ml729)

    # tmp32: ml732, full, tmp31: ml729, full
    ml733 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml729, ml733, info) = LinearAlgebra.LAPACK.getrf!(ml729)

    # tmp32: ml732, full, P35: ml733, ipiv, L33: ml729, lower_triangular_udiag, U34: ml729, upper_triangular
    ml734 = [1:length(ml733);]
    @inbounds for i in 1:length(ml733)
        ml734[i], ml734[ml733[i]] = ml734[ml733[i]], ml734[i];
    end;
    ml735 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml735 = ml732[ml734]

    # L33: ml729, lower_triangular_udiag, U34: ml729, upper_triangular, tmp40: ml735, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml729, ml735)

    # U34: ml729, upper_triangular, tmp41: ml735, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml729, ml735)

    # tmp17: ml735, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml735), (finish-start)*1e-9)
end