using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm17(ml570::Array{Float64,2}, ml571::Array{Float64,2}, ml572::Array{Float64,2}, ml573::Array{Float64,2}, ml574::Array{Float64,1})
    # cost 5.07e+10
    # R: ml570, full, L: ml571, full, A: ml572, full, B: ml573, full, y: ml574, full
    ml575 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml575, ml573)

    # R: ml570, full, L: ml571, full, A: ml572, full, y: ml574, full, tmp26: ml575, full
    ml576 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml572, ml576, info) = LinearAlgebra.LAPACK.getrf!(ml572)

    # R: ml570, full, L: ml571, full, y: ml574, full, tmp26: ml575, full, P11: ml576, ipiv, L9: ml572, lower_triangular_udiag, U10: ml572, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml572, ml575)

    # R: ml570, full, L: ml571, full, y: ml574, full, P11: ml576, ipiv, L9: ml572, lower_triangular_udiag, tmp27: ml575, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml572, ml575)

    # R: ml570, full, L: ml571, full, y: ml574, full, P11: ml576, ipiv, tmp28: ml575, full
    ml577 = [1:length(ml576);]
    @inbounds for i in 1:length(ml576)
        ml577[i], ml577[ml576[i]] = ml577[ml576[i]], ml577[i];
    end;
    ml578 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml578 = ml575[invperm(ml577),:]

    # R: ml570, full, L: ml571, full, y: ml574, full, tmp25: ml578, full
    ml579 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml578, 0.0, ml579)

    # R: ml570, full, L: ml571, full, y: ml574, full, tmp19: ml579, symmetric_lower_triangular
    ml580 = diag(ml571)
    ml581 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml570, 1, ml581, 1)
    # tmp29 = (L R)
    for i = 1:size(ml570, 2);
        view(ml570, :, i)[:] .*= ml580;
    end;        

    # R: ml581, full, y: ml574, full, tmp19: ml579, symmetric_lower_triangular, tmp29: ml570, full
    for i = 1:2000-1;
        view(ml579, i, i+1:2000)[:] = view(ml579, i+1:2000, i);
    end;
    ml582 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml579, 1, ml582, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml570, ml581, 1.0, ml579)

    # y: ml574, full, tmp19: ml582, full, tmp31: ml579, full
    ml583 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml579, ml583, info) = LinearAlgebra.LAPACK.getrf!(ml579)

    # y: ml574, full, tmp19: ml582, full, P35: ml583, ipiv, L33: ml579, lower_triangular_udiag, U34: ml579, upper_triangular
    ml584 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml582, ml574, 0.0, ml584)

    # P35: ml583, ipiv, L33: ml579, lower_triangular_udiag, U34: ml579, upper_triangular, tmp32: ml584, full
    ml585 = [1:length(ml583);]
    @inbounds for i in 1:length(ml583)
        ml585[i], ml585[ml583[i]] = ml585[ml583[i]], ml585[i];
    end;
    ml586 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml586 = ml584[ml585]

    # L33: ml579, lower_triangular_udiag, U34: ml579, upper_triangular, tmp40: ml586, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml579, ml586)

    # U34: ml579, upper_triangular, tmp41: ml586, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml579, ml586)

    # tmp17: ml586, full
    # x = tmp17
    return (ml586)
end