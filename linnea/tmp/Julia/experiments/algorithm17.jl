using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm17(ml587::Array{Float64,2}, ml588::Array{Float64,2}, ml589::Array{Float64,2}, ml590::Array{Float64,2}, ml591::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml587, full, L: ml588, full, A: ml589, full, B: ml590, full, y: ml591, full
    ml592 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml592, ml590)

    # R: ml587, full, L: ml588, full, A: ml589, full, y: ml591, full, tmp26: ml592, full
    ml593 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml589, ml593, info) = LinearAlgebra.LAPACK.getrf!(ml589)

    # R: ml587, full, L: ml588, full, y: ml591, full, tmp26: ml592, full, P11: ml593, ipiv, L9: ml589, lower_triangular_udiag, U10: ml589, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml589, ml592)

    # R: ml587, full, L: ml588, full, y: ml591, full, P11: ml593, ipiv, L9: ml589, lower_triangular_udiag, tmp27: ml592, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml589, ml592)

    # R: ml587, full, L: ml588, full, y: ml591, full, P11: ml593, ipiv, tmp28: ml592, full
    ml594 = [1:length(ml593);]
    @inbounds for i in 1:length(ml593)
        ml594[i], ml594[ml593[i]] = ml594[ml593[i]], ml594[i];
    end;
    ml595 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml595 = ml592[invperm(ml594),:]

    # R: ml587, full, L: ml588, full, y: ml591, full, tmp25: ml595, full
    ml596 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml595, 0.0, ml596)

    # R: ml587, full, L: ml588, full, y: ml591, full, tmp19: ml596, symmetric_lower_triangular
    ml597 = diag(ml588)
    ml598 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml587, 1, ml598, 1)
    # tmp29 = (L R)
    for i = 1:size(ml587, 2);
        view(ml587, :, i)[:] .*= ml597;
    end;        

    # R: ml598, full, y: ml591, full, tmp19: ml596, symmetric_lower_triangular, tmp29: ml587, full
    for i = 1:2000-1;
        view(ml596, i, i+1:2000)[:] = view(ml596, i+1:2000, i);
    end;
    ml599 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml596, 1, ml599, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml587, ml598, 1.0, ml596)

    # y: ml591, full, tmp19: ml599, full, tmp31: ml596, full
    ml600 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml596, ml600, info) = LinearAlgebra.LAPACK.getrf!(ml596)

    # y: ml591, full, tmp19: ml599, full, P35: ml600, ipiv, L33: ml596, lower_triangular_udiag, U34: ml596, upper_triangular
    ml601 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml599, ml591, 0.0, ml601)

    # P35: ml600, ipiv, L33: ml596, lower_triangular_udiag, U34: ml596, upper_triangular, tmp32: ml601, full
    ml602 = [1:length(ml600);]
    @inbounds for i in 1:length(ml600)
        ml602[i], ml602[ml600[i]] = ml602[ml600[i]], ml602[i];
    end;
    ml603 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml603 = ml601[ml602]

    # L33: ml596, lower_triangular_udiag, U34: ml596, upper_triangular, tmp40: ml603, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml596, ml603)

    # U34: ml596, upper_triangular, tmp41: ml603, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml596, ml603)

    # tmp17: ml603, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml603), (finish-start)*1e-9)
end