using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm20(ml688::Array{Float64,2}, ml689::Array{Float64,2}, ml690::Array{Float64,2}, ml691::Array{Float64,2}, ml692::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml688, full, L: ml689, full, A: ml690, full, B: ml691, full, y: ml692, full
    ml693 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml690, ml693, info) = LinearAlgebra.LAPACK.getrf!(ml690)

    # R: ml688, full, L: ml689, full, B: ml691, full, y: ml692, full, P11: ml693, ipiv, L9: ml690, lower_triangular_udiag, U10: ml690, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml690, ml691)

    # R: ml688, full, L: ml689, full, y: ml692, full, P11: ml693, ipiv, L9: ml690, lower_triangular_udiag, tmp53: ml691, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml690, ml691)

    # R: ml688, full, L: ml689, full, y: ml692, full, P11: ml693, ipiv, tmp54: ml691, full
    ml694 = [1:length(ml693);]
    @inbounds for i in 1:length(ml693)
        ml694[i], ml694[ml693[i]] = ml694[ml693[i]], ml694[i];
    end;
    ml695 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml695 = ml691[:,invperm(ml694)]

    # R: ml688, full, L: ml689, full, y: ml692, full, tmp55: ml695, full
    ml696 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml696, ml695)

    # R: ml688, full, L: ml689, full, y: ml692, full, tmp25: ml696, full
    ml697 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml696, 0.0, ml697)

    # R: ml688, full, L: ml689, full, y: ml692, full, tmp19: ml697, symmetric_lower_triangular
    ml698 = diag(ml689)
    ml699 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml688, 1, ml699, 1)
    # tmp29 = (L R)
    for i = 1:size(ml688, 2);
        view(ml688, :, i)[:] .*= ml698;
    end;        

    # R: ml699, full, y: ml692, full, tmp19: ml697, symmetric_lower_triangular, tmp29: ml688, full
    ml700 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml697, ml692, 0.0, ml700)

    # R: ml699, full, tmp19: ml697, symmetric_lower_triangular, tmp29: ml688, full, tmp32: ml700, full
    for i = 1:2000-1;
        view(ml697, i, i+1:2000)[:] = view(ml697, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml688, ml699, 1.0, ml697)

    # tmp32: ml700, full, tmp31: ml697, full
    ml701 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml697, ml701, info) = LinearAlgebra.LAPACK.getrf!(ml697)

    # tmp32: ml700, full, P35: ml701, ipiv, L33: ml697, lower_triangular_udiag, U34: ml697, upper_triangular
    ml702 = [1:length(ml701);]
    @inbounds for i in 1:length(ml701)
        ml702[i], ml702[ml701[i]] = ml702[ml701[i]], ml702[i];
    end;
    ml703 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml703 = ml700[ml702]

    # L33: ml697, lower_triangular_udiag, U34: ml697, upper_triangular, tmp40: ml703, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml697, ml703)

    # U34: ml697, upper_triangular, tmp41: ml703, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml697, ml703)

    # tmp17: ml703, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml703), (finish-start)*1e-9)
end