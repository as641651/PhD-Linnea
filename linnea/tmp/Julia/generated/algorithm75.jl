using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm75(ml2510::Array{Float64,2}, ml2511::Array{Float64,2}, ml2512::Array{Float64,2}, ml2513::Array{Float64,2}, ml2514::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2510, full, L: ml2511, full, A: ml2512, full, B: ml2513, full, y: ml2514, full
    ml2515 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2515, ml2513)

    # R: ml2510, full, L: ml2511, full, A: ml2512, full, y: ml2514, full, tmp26: ml2515, full
    ml2516 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2512, ml2516, info) = LinearAlgebra.LAPACK.getrf!(ml2512)

    # R: ml2510, full, L: ml2511, full, y: ml2514, full, tmp26: ml2515, full, P11: ml2516, ipiv, L9: ml2512, lower_triangular_udiag, U10: ml2512, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2512, ml2515)

    # R: ml2510, full, L: ml2511, full, y: ml2514, full, P11: ml2516, ipiv, L9: ml2512, lower_triangular_udiag, tmp27: ml2515, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2512, ml2515)

    # R: ml2510, full, L: ml2511, full, y: ml2514, full, P11: ml2516, ipiv, tmp28: ml2515, full
    ml2517 = [1:length(ml2516);]
    @inbounds for i in 1:length(ml2516)
        ml2517[i], ml2517[ml2516[i]] = ml2517[ml2516[i]], ml2517[i];
    end;
    ml2518 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2518 = ml2515[invperm(ml2517),:]

    # R: ml2510, full, L: ml2511, full, y: ml2514, full, tmp25: ml2518, full
    ml2519 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2518, 0.0, ml2519)

    # R: ml2510, full, L: ml2511, full, y: ml2514, full, tmp19: ml2519, symmetric_lower_triangular
    ml2520 = diag(ml2511)
    ml2521 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2510, 1, ml2521, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2510, 2);
        view(ml2510, :, i)[:] .*= ml2520;
    end;        

    # R: ml2521, full, y: ml2514, full, tmp19: ml2519, symmetric_lower_triangular, tmp29: ml2510, full
    for i = 1:2000-1;
        view(ml2519, i, i+1:2000)[:] = view(ml2519, i+1:2000, i);
    end;
    ml2522 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2519, 1, ml2522, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2510, ml2521, 1.0, ml2519)

    # y: ml2514, full, tmp19: ml2522, full, tmp31: ml2519, full
    ml2523 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2519, ml2523, info) = LinearAlgebra.LAPACK.getrf!(ml2519)

    # y: ml2514, full, tmp19: ml2522, full, P35: ml2523, ipiv, L33: ml2519, lower_triangular_udiag, U34: ml2519, upper_triangular
    ml2524 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2522, ml2514, 0.0, ml2524)

    # P35: ml2523, ipiv, L33: ml2519, lower_triangular_udiag, U34: ml2519, upper_triangular, tmp32: ml2524, full
    ml2525 = [1:length(ml2523);]
    @inbounds for i in 1:length(ml2523)
        ml2525[i], ml2525[ml2523[i]] = ml2525[ml2523[i]], ml2525[i];
    end;
    ml2526 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2526 = ml2524[ml2525]

    # L33: ml2519, lower_triangular_udiag, U34: ml2519, upper_triangular, tmp40: ml2526, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2519, ml2526)

    # U34: ml2519, upper_triangular, tmp41: ml2526, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2519, ml2526)

    # tmp17: ml2526, full
    # x = tmp17
    return (ml2526)
end