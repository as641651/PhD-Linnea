using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm59(ml1991::Array{Float64,2}, ml1992::Array{Float64,2}, ml1993::Array{Float64,2}, ml1994::Array{Float64,2}, ml1995::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1991, full, L: ml1992, full, A: ml1993, full, B: ml1994, full, y: ml1995, full
    ml1996 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1993, ml1996, info) = LinearAlgebra.LAPACK.getrf!(ml1993)

    # R: ml1991, full, L: ml1992, full, B: ml1994, full, y: ml1995, full, P11: ml1996, ipiv, L9: ml1993, lower_triangular_udiag, U10: ml1993, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1993, ml1994)

    # R: ml1991, full, L: ml1992, full, y: ml1995, full, P11: ml1996, ipiv, L9: ml1993, lower_triangular_udiag, tmp53: ml1994, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1993, ml1994)

    # R: ml1991, full, L: ml1992, full, y: ml1995, full, P11: ml1996, ipiv, tmp54: ml1994, full
    ml1997 = [1:length(ml1996);]
    @inbounds for i in 1:length(ml1996)
        ml1997[i], ml1997[ml1996[i]] = ml1997[ml1996[i]], ml1997[i];
    end;
    ml1998 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1998 = ml1994[:,invperm(ml1997)]

    # R: ml1991, full, L: ml1992, full, y: ml1995, full, tmp55: ml1998, full
    ml1999 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1999, ml1998)

    # R: ml1991, full, L: ml1992, full, y: ml1995, full, tmp25: ml1999, full
    ml2000 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1999, 0.0, ml2000)

    # R: ml1991, full, L: ml1992, full, y: ml1995, full, tmp19: ml2000, symmetric_lower_triangular
    ml2001 = diag(ml1992)
    ml2002 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1991, 1, ml2002, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1991, 2);
        view(ml1991, :, i)[:] .*= ml2001;
    end;        

    # R: ml2002, full, y: ml1995, full, tmp19: ml2000, symmetric_lower_triangular, tmp29: ml1991, full
    for i = 1:2000-1;
        view(ml2000, i, i+1:2000)[:] = view(ml2000, i+1:2000, i);
    end;
    ml2003 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2000, 1, ml2003, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1991, ml2002, 1.0, ml2000)

    # y: ml1995, full, tmp19: ml2003, full, tmp31: ml2000, full
    ml2004 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2000, ml2004, info) = LinearAlgebra.LAPACK.getrf!(ml2000)

    # y: ml1995, full, tmp19: ml2003, full, P35: ml2004, ipiv, L33: ml2000, lower_triangular_udiag, U34: ml2000, upper_triangular
    ml2005 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2003, ml1995, 0.0, ml2005)

    # P35: ml2004, ipiv, L33: ml2000, lower_triangular_udiag, U34: ml2000, upper_triangular, tmp32: ml2005, full
    ml2006 = [1:length(ml2004);]
    @inbounds for i in 1:length(ml2004)
        ml2006[i], ml2006[ml2004[i]] = ml2006[ml2004[i]], ml2006[i];
    end;
    ml2007 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2007 = ml2005[ml2006]

    # L33: ml2000, lower_triangular_udiag, U34: ml2000, upper_triangular, tmp40: ml2007, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2000, ml2007)

    # U34: ml2000, upper_triangular, tmp41: ml2007, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2000, ml2007)

    # tmp17: ml2007, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml2007), (finish-start)*1e-9)
end