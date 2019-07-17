using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm2(ml80::Array{Float64,2}, ml81::Array{Float64,2}, ml82::Array{Float64,2}, ml83::Array{Float64,2}, ml84::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml80, full, L: ml81, full, A: ml82, full, B: ml83, full, y: ml84, full
    ml85 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml82, ml85, info) = LinearAlgebra.LAPACK.getrf!(ml82)

    # R: ml80, full, L: ml81, full, B: ml83, full, y: ml84, full, P11: ml85, ipiv, L9: ml82, lower_triangular_udiag, U10: ml82, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml82, ml83)

    # R: ml80, full, L: ml81, full, y: ml84, full, P11: ml85, ipiv, L9: ml82, lower_triangular_udiag, tmp53: ml83, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml82, ml83)

    # R: ml80, full, L: ml81, full, y: ml84, full, P11: ml85, ipiv, tmp54: ml83, full
    ml86 = [1:length(ml85);]
    @inbounds for i in 1:length(ml85)
        ml86[i], ml86[ml85[i]] = ml86[ml85[i]], ml86[i];
    end;
    ml87 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml87 = ml83[:,invperm(ml86)]

    # R: ml80, full, L: ml81, full, y: ml84, full, tmp55: ml87, full
    ml88 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml88, ml87)

    # R: ml80, full, L: ml81, full, y: ml84, full, tmp25: ml88, full
    ml89 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml88, 0.0, ml89)

    # R: ml80, full, L: ml81, full, y: ml84, full, tmp19: ml89, symmetric_lower_triangular
    ml90 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml89, ml84, 0.0, ml90)

    # R: ml80, full, L: ml81, full, tmp19: ml89, symmetric_lower_triangular, tmp32: ml90, full
    ml91 = diag(ml81)
    ml92 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml80, 1, ml92, 1)
    # tmp29 = (L R)
    for i = 1:size(ml80, 2);
        view(ml80, :, i)[:] .*= ml91;
    end;        

    # R: ml92, full, tmp19: ml89, symmetric_lower_triangular, tmp32: ml90, full, tmp29: ml80, full
    for i = 1:2000-1;
        view(ml89, i, i+1:2000)[:] = view(ml89, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml92, ml80, 1.0, ml89)

    # tmp32: ml90, full, tmp31: ml89, full
    ml93 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml89, ml93, info) = LinearAlgebra.LAPACK.getrf!(ml89)

    # tmp32: ml90, full, P35: ml93, ipiv, L33: ml89, lower_triangular_udiag, U34: ml89, upper_triangular
    ml94 = [1:length(ml93);]
    @inbounds for i in 1:length(ml93)
        ml94[i], ml94[ml93[i]] = ml94[ml93[i]], ml94[i];
    end;
    ml95 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml95 = ml90[ml94]

    # L33: ml89, lower_triangular_udiag, U34: ml89, upper_triangular, tmp40: ml95, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml89, ml95)

    # U34: ml89, upper_triangular, tmp41: ml95, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml89, ml95)

    # tmp17: ml95, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml95), (finish-start)*1e-9)
end