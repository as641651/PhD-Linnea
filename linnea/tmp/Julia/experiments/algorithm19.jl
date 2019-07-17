using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm19(ml655::Array{Float64,2}, ml656::Array{Float64,2}, ml657::Array{Float64,2}, ml658::Array{Float64,2}, ml659::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml655, full, L: ml656, full, A: ml657, full, B: ml658, full, y: ml659, full
    ml660 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml660, ml658)

    # R: ml655, full, L: ml656, full, A: ml657, full, y: ml659, full, tmp26: ml660, full
    ml661 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml657, ml661, info) = LinearAlgebra.LAPACK.getrf!(ml657)

    # R: ml655, full, L: ml656, full, y: ml659, full, tmp26: ml660, full, P11: ml661, ipiv, L9: ml657, lower_triangular_udiag, U10: ml657, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml657, ml660)

    # R: ml655, full, L: ml656, full, y: ml659, full, P11: ml661, ipiv, L9: ml657, lower_triangular_udiag, tmp27: ml660, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml657, ml660)

    # R: ml655, full, L: ml656, full, y: ml659, full, P11: ml661, ipiv, tmp28: ml660, full
    ml662 = [1:length(ml661);]
    @inbounds for i in 1:length(ml661)
        ml662[i], ml662[ml661[i]] = ml662[ml661[i]], ml662[i];
    end;
    ml663 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml663 = ml660[invperm(ml662),:]

    # R: ml655, full, L: ml656, full, y: ml659, full, tmp25: ml663, full
    ml664 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml663, 0.0, ml664)

    # R: ml655, full, L: ml656, full, y: ml659, full, tmp19: ml664, symmetric_lower_triangular
    ml665 = diag(ml656)
    ml666 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml655, 1, ml666, 1)
    # tmp29 = (L R)
    for i = 1:size(ml655, 2);
        view(ml655, :, i)[:] .*= ml665;
    end;        

    # R: ml666, full, y: ml659, full, tmp19: ml664, symmetric_lower_triangular, tmp29: ml655, full
    for i = 1:2000-1;
        view(ml664, i, i+1:2000)[:] = view(ml664, i+1:2000, i);
    end;
    ml667 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml664, 1, ml667, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml655, ml666, 1.0, ml664)

    # y: ml659, full, tmp19: ml667, full, tmp31: ml664, full
    ml668 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml664, ml668, info) = LinearAlgebra.LAPACK.getrf!(ml664)

    # y: ml659, full, tmp19: ml667, full, P35: ml668, ipiv, L33: ml664, lower_triangular_udiag, U34: ml664, upper_triangular
    ml669 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml667, ml659, 0.0, ml669)

    # P35: ml668, ipiv, L33: ml664, lower_triangular_udiag, U34: ml664, upper_triangular, tmp32: ml669, full
    ml670 = [1:length(ml668);]
    @inbounds for i in 1:length(ml668)
        ml670[i], ml670[ml668[i]] = ml670[ml668[i]], ml670[i];
    end;
    ml671 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml671 = ml669[ml670]

    # L33: ml664, lower_triangular_udiag, U34: ml664, upper_triangular, tmp40: ml671, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml664, ml671)

    # U34: ml664, upper_triangular, tmp41: ml671, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml664, ml671)

    # tmp17: ml671, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml671), (finish-start)*1e-9)
end