using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm78(ml2612::Array{Float64,2}, ml2613::Array{Float64,2}, ml2614::Array{Float64,2}, ml2615::Array{Float64,2}, ml2616::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2612, full, L: ml2613, full, A: ml2614, full, B: ml2615, full, y: ml2616, full
    ml2617 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2617, ml2615)

    # R: ml2612, full, L: ml2613, full, A: ml2614, full, y: ml2616, full, tmp26: ml2617, full
    ml2618 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2614, ml2618, info) = LinearAlgebra.LAPACK.getrf!(ml2614)

    # R: ml2612, full, L: ml2613, full, y: ml2616, full, tmp26: ml2617, full, P11: ml2618, ipiv, L9: ml2614, lower_triangular_udiag, U10: ml2614, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2614, ml2617)

    # R: ml2612, full, L: ml2613, full, y: ml2616, full, P11: ml2618, ipiv, L9: ml2614, lower_triangular_udiag, tmp27: ml2617, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2614, ml2617)

    # R: ml2612, full, L: ml2613, full, y: ml2616, full, P11: ml2618, ipiv, tmp28: ml2617, full
    ml2619 = [1:length(ml2618);]
    @inbounds for i in 1:length(ml2618)
        ml2619[i], ml2619[ml2618[i]] = ml2619[ml2618[i]], ml2619[i];
    end;
    ml2620 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2620 = ml2617[invperm(ml2619),:]

    # R: ml2612, full, L: ml2613, full, y: ml2616, full, tmp25: ml2620, full
    ml2621 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2620, 0.0, ml2621)

    # R: ml2612, full, L: ml2613, full, y: ml2616, full, tmp19: ml2621, symmetric_lower_triangular
    ml2622 = diag(ml2613)
    ml2623 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2612, 1, ml2623, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2612, 2);
        view(ml2612, :, i)[:] .*= ml2622;
    end;        

    # R: ml2623, full, y: ml2616, full, tmp19: ml2621, symmetric_lower_triangular, tmp29: ml2612, full
    for i = 1:2000-1;
        view(ml2621, i, i+1:2000)[:] = view(ml2621, i+1:2000, i);
    end;
    ml2624 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2621, 1, ml2624, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2612, ml2623, 1.0, ml2621)

    # y: ml2616, full, tmp19: ml2624, full, tmp31: ml2621, full
    ml2625 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2621, ml2625, info) = LinearAlgebra.LAPACK.getrf!(ml2621)

    # y: ml2616, full, tmp19: ml2624, full, P35: ml2625, ipiv, L33: ml2621, lower_triangular_udiag, U34: ml2621, upper_triangular
    ml2626 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2624, ml2616, 0.0, ml2626)

    # P35: ml2625, ipiv, L33: ml2621, lower_triangular_udiag, U34: ml2621, upper_triangular, tmp32: ml2626, full
    ml2627 = [1:length(ml2625);]
    @inbounds for i in 1:length(ml2625)
        ml2627[i], ml2627[ml2625[i]] = ml2627[ml2625[i]], ml2627[i];
    end;
    ml2628 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2628 = ml2626[ml2627]

    # L33: ml2621, lower_triangular_udiag, U34: ml2621, upper_triangular, tmp40: ml2628, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2621, ml2628)

    # U34: ml2621, upper_triangular, tmp41: ml2628, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2621, ml2628)

    # tmp17: ml2628, full
    # x = tmp17
    return (ml2628)
end