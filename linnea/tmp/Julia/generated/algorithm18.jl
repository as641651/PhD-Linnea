using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm18(ml604::Array{Float64,2}, ml605::Array{Float64,2}, ml606::Array{Float64,2}, ml607::Array{Float64,2}, ml608::Array{Float64,1})
    # cost 5.07e+10
    # R: ml604, full, L: ml605, full, A: ml606, full, B: ml607, full, y: ml608, full
    ml609 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml609, ml607)

    # R: ml604, full, L: ml605, full, A: ml606, full, y: ml608, full, tmp26: ml609, full
    ml610 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml606, ml610, info) = LinearAlgebra.LAPACK.getrf!(ml606)

    # R: ml604, full, L: ml605, full, y: ml608, full, tmp26: ml609, full, P11: ml610, ipiv, L9: ml606, lower_triangular_udiag, U10: ml606, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml606, ml609)

    # R: ml604, full, L: ml605, full, y: ml608, full, P11: ml610, ipiv, L9: ml606, lower_triangular_udiag, tmp27: ml609, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml606, ml609)

    # R: ml604, full, L: ml605, full, y: ml608, full, P11: ml610, ipiv, tmp28: ml609, full
    ml611 = [1:length(ml610);]
    @inbounds for i in 1:length(ml610)
        ml611[i], ml611[ml610[i]] = ml611[ml610[i]], ml611[i];
    end;
    ml612 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml612 = ml609[invperm(ml611),:]

    # R: ml604, full, L: ml605, full, y: ml608, full, tmp25: ml612, full
    ml613 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml612, 0.0, ml613)

    # R: ml604, full, L: ml605, full, y: ml608, full, tmp19: ml613, symmetric_lower_triangular
    ml614 = diag(ml605)
    ml615 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml604, 1, ml615, 1)
    # tmp29 = (L R)
    for i = 1:size(ml604, 2);
        view(ml604, :, i)[:] .*= ml614;
    end;        

    # R: ml615, full, y: ml608, full, tmp19: ml613, symmetric_lower_triangular, tmp29: ml604, full
    for i = 1:2000-1;
        view(ml613, i, i+1:2000)[:] = view(ml613, i+1:2000, i);
    end;
    ml616 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml613, 1, ml616, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml604, ml615, 1.0, ml613)

    # y: ml608, full, tmp19: ml616, full, tmp31: ml613, full
    ml617 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml613, ml617, info) = LinearAlgebra.LAPACK.getrf!(ml613)

    # y: ml608, full, tmp19: ml616, full, P35: ml617, ipiv, L33: ml613, lower_triangular_udiag, U34: ml613, upper_triangular
    ml618 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml616, ml608, 0.0, ml618)

    # P35: ml617, ipiv, L33: ml613, lower_triangular_udiag, U34: ml613, upper_triangular, tmp32: ml618, full
    ml619 = [1:length(ml617);]
    @inbounds for i in 1:length(ml617)
        ml619[i], ml619[ml617[i]] = ml619[ml617[i]], ml619[i];
    end;
    ml620 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml620 = ml618[ml619]

    # L33: ml613, lower_triangular_udiag, U34: ml613, upper_triangular, tmp40: ml620, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml613, ml620)

    # U34: ml613, upper_triangular, tmp41: ml620, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml613, ml620)

    # tmp17: ml620, full
    # x = tmp17
    return (ml620)
end