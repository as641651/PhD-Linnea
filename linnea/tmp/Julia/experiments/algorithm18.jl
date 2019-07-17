using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm18(ml621::Array{Float64,2}, ml622::Array{Float64,2}, ml623::Array{Float64,2}, ml624::Array{Float64,2}, ml625::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml621, full, L: ml622, full, A: ml623, full, B: ml624, full, y: ml625, full
    ml626 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml626, ml624)

    # R: ml621, full, L: ml622, full, A: ml623, full, y: ml625, full, tmp26: ml626, full
    ml627 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml623, ml627, info) = LinearAlgebra.LAPACK.getrf!(ml623)

    # R: ml621, full, L: ml622, full, y: ml625, full, tmp26: ml626, full, P11: ml627, ipiv, L9: ml623, lower_triangular_udiag, U10: ml623, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml623, ml626)

    # R: ml621, full, L: ml622, full, y: ml625, full, P11: ml627, ipiv, L9: ml623, lower_triangular_udiag, tmp27: ml626, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml623, ml626)

    # R: ml621, full, L: ml622, full, y: ml625, full, P11: ml627, ipiv, tmp28: ml626, full
    ml628 = [1:length(ml627);]
    @inbounds for i in 1:length(ml627)
        ml628[i], ml628[ml627[i]] = ml628[ml627[i]], ml628[i];
    end;
    ml629 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml629 = ml626[invperm(ml628),:]

    # R: ml621, full, L: ml622, full, y: ml625, full, tmp25: ml629, full
    ml630 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml629, 0.0, ml630)

    # R: ml621, full, L: ml622, full, y: ml625, full, tmp19: ml630, symmetric_lower_triangular
    ml631 = diag(ml622)
    ml632 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml621, 1, ml632, 1)
    # tmp29 = (L R)
    for i = 1:size(ml621, 2);
        view(ml621, :, i)[:] .*= ml631;
    end;        

    # R: ml632, full, y: ml625, full, tmp19: ml630, symmetric_lower_triangular, tmp29: ml621, full
    for i = 1:2000-1;
        view(ml630, i, i+1:2000)[:] = view(ml630, i+1:2000, i);
    end;
    ml633 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml630, 1, ml633, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml621, ml632, 1.0, ml630)

    # y: ml625, full, tmp19: ml633, full, tmp31: ml630, full
    ml634 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml630, ml634, info) = LinearAlgebra.LAPACK.getrf!(ml630)

    # y: ml625, full, tmp19: ml633, full, P35: ml634, ipiv, L33: ml630, lower_triangular_udiag, U34: ml630, upper_triangular
    ml635 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml633, ml625, 0.0, ml635)

    # P35: ml634, ipiv, L33: ml630, lower_triangular_udiag, U34: ml630, upper_triangular, tmp32: ml635, full
    ml636 = [1:length(ml634);]
    @inbounds for i in 1:length(ml634)
        ml636[i], ml636[ml634[i]] = ml636[ml634[i]], ml636[i];
    end;
    ml637 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml637 = ml635[ml636]

    # L33: ml630, lower_triangular_udiag, U34: ml630, upper_triangular, tmp40: ml637, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml630, ml637)

    # U34: ml630, upper_triangular, tmp41: ml637, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml630, ml637)

    # tmp17: ml637, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml637), (finish-start)*1e-9)
end