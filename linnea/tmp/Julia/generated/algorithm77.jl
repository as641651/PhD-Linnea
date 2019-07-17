using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm77(ml2578::Array{Float64,2}, ml2579::Array{Float64,2}, ml2580::Array{Float64,2}, ml2581::Array{Float64,2}, ml2582::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2578, full, L: ml2579, full, A: ml2580, full, B: ml2581, full, y: ml2582, full
    ml2583 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2583, ml2581)

    # R: ml2578, full, L: ml2579, full, A: ml2580, full, y: ml2582, full, tmp26: ml2583, full
    ml2584 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2580, ml2584, info) = LinearAlgebra.LAPACK.getrf!(ml2580)

    # R: ml2578, full, L: ml2579, full, y: ml2582, full, tmp26: ml2583, full, P11: ml2584, ipiv, L9: ml2580, lower_triangular_udiag, U10: ml2580, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2580, ml2583)

    # R: ml2578, full, L: ml2579, full, y: ml2582, full, P11: ml2584, ipiv, L9: ml2580, lower_triangular_udiag, tmp27: ml2583, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2580, ml2583)

    # R: ml2578, full, L: ml2579, full, y: ml2582, full, P11: ml2584, ipiv, tmp28: ml2583, full
    ml2585 = [1:length(ml2584);]
    @inbounds for i in 1:length(ml2584)
        ml2585[i], ml2585[ml2584[i]] = ml2585[ml2584[i]], ml2585[i];
    end;
    ml2586 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2586 = ml2583[invperm(ml2585),:]

    # R: ml2578, full, L: ml2579, full, y: ml2582, full, tmp25: ml2586, full
    ml2587 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2586, 0.0, ml2587)

    # R: ml2578, full, L: ml2579, full, y: ml2582, full, tmp19: ml2587, symmetric_lower_triangular
    ml2588 = diag(ml2579)
    ml2589 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2578, 1, ml2589, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2578, 2);
        view(ml2578, :, i)[:] .*= ml2588;
    end;        

    # R: ml2589, full, y: ml2582, full, tmp19: ml2587, symmetric_lower_triangular, tmp29: ml2578, full
    for i = 1:2000-1;
        view(ml2587, i, i+1:2000)[:] = view(ml2587, i+1:2000, i);
    end;
    ml2590 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2587, 1, ml2590, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2578, ml2589, 1.0, ml2587)

    # y: ml2582, full, tmp19: ml2590, full, tmp31: ml2587, full
    ml2591 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2587, ml2591, info) = LinearAlgebra.LAPACK.getrf!(ml2587)

    # y: ml2582, full, tmp19: ml2590, full, P35: ml2591, ipiv, L33: ml2587, lower_triangular_udiag, U34: ml2587, upper_triangular
    ml2592 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2590, ml2582, 0.0, ml2592)

    # P35: ml2591, ipiv, L33: ml2587, lower_triangular_udiag, U34: ml2587, upper_triangular, tmp32: ml2592, full
    ml2593 = [1:length(ml2591);]
    @inbounds for i in 1:length(ml2591)
        ml2593[i], ml2593[ml2591[i]] = ml2593[ml2591[i]], ml2593[i];
    end;
    ml2594 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2594 = ml2592[ml2593]

    # L33: ml2587, lower_triangular_udiag, U34: ml2587, upper_triangular, tmp40: ml2594, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2587, ml2594)

    # U34: ml2587, upper_triangular, tmp41: ml2594, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2587, ml2594)

    # tmp17: ml2594, full
    # x = tmp17
    return (ml2594)
end