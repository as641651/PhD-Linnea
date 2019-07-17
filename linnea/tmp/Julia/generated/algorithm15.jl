using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm15(ml502::Array{Float64,2}, ml503::Array{Float64,2}, ml504::Array{Float64,2}, ml505::Array{Float64,2}, ml506::Array{Float64,1})
    # cost 5.07e+10
    # R: ml502, full, L: ml503, full, A: ml504, full, B: ml505, full, y: ml506, full
    ml507 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml507, ml505)

    # R: ml502, full, L: ml503, full, A: ml504, full, y: ml506, full, tmp26: ml507, full
    ml508 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml504, ml508, info) = LinearAlgebra.LAPACK.getrf!(ml504)

    # R: ml502, full, L: ml503, full, y: ml506, full, tmp26: ml507, full, P11: ml508, ipiv, L9: ml504, lower_triangular_udiag, U10: ml504, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml504, ml507)

    # R: ml502, full, L: ml503, full, y: ml506, full, P11: ml508, ipiv, L9: ml504, lower_triangular_udiag, tmp27: ml507, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml504, ml507)

    # R: ml502, full, L: ml503, full, y: ml506, full, P11: ml508, ipiv, tmp28: ml507, full
    ml509 = [1:length(ml508);]
    @inbounds for i in 1:length(ml508)
        ml509[i], ml509[ml508[i]] = ml509[ml508[i]], ml509[i];
    end;
    ml510 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml510 = ml507[invperm(ml509),:]

    # R: ml502, full, L: ml503, full, y: ml506, full, tmp25: ml510, full
    ml511 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml510, 0.0, ml511)

    # R: ml502, full, L: ml503, full, y: ml506, full, tmp19: ml511, symmetric_lower_triangular
    ml512 = diag(ml503)
    ml513 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml502, 1, ml513, 1)
    # tmp29 = (L R)
    for i = 1:size(ml502, 2);
        view(ml502, :, i)[:] .*= ml512;
    end;        

    # R: ml513, full, y: ml506, full, tmp19: ml511, symmetric_lower_triangular, tmp29: ml502, full
    for i = 1:2000-1;
        view(ml511, i, i+1:2000)[:] = view(ml511, i+1:2000, i);
    end;
    ml514 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml511, 1, ml514, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml502, ml513, 1.0, ml511)

    # y: ml506, full, tmp19: ml514, full, tmp31: ml511, full
    ml515 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml511, ml515, info) = LinearAlgebra.LAPACK.getrf!(ml511)

    # y: ml506, full, tmp19: ml514, full, P35: ml515, ipiv, L33: ml511, lower_triangular_udiag, U34: ml511, upper_triangular
    ml516 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml514, ml506, 0.0, ml516)

    # P35: ml515, ipiv, L33: ml511, lower_triangular_udiag, U34: ml511, upper_triangular, tmp32: ml516, full
    ml517 = [1:length(ml515);]
    @inbounds for i in 1:length(ml515)
        ml517[i], ml517[ml515[i]] = ml517[ml515[i]], ml517[i];
    end;
    ml518 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml518 = ml516[ml517]

    # L33: ml511, lower_triangular_udiag, U34: ml511, upper_triangular, tmp40: ml518, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml511, ml518)

    # U34: ml511, upper_triangular, tmp41: ml518, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml511, ml518)

    # tmp17: ml518, full
    # x = tmp17
    return (ml518)
end