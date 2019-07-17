using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm14(ml468::Array{Float64,2}, ml469::Array{Float64,2}, ml470::Array{Float64,2}, ml471::Array{Float64,2}, ml472::Array{Float64,1})
    # cost 5.07e+10
    # R: ml468, full, L: ml469, full, A: ml470, full, B: ml471, full, y: ml472, full
    ml473 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml473, ml471)

    # R: ml468, full, L: ml469, full, A: ml470, full, y: ml472, full, tmp26: ml473, full
    ml474 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml470, ml474, info) = LinearAlgebra.LAPACK.getrf!(ml470)

    # R: ml468, full, L: ml469, full, y: ml472, full, tmp26: ml473, full, P11: ml474, ipiv, L9: ml470, lower_triangular_udiag, U10: ml470, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml470, ml473)

    # R: ml468, full, L: ml469, full, y: ml472, full, P11: ml474, ipiv, L9: ml470, lower_triangular_udiag, tmp27: ml473, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml470, ml473)

    # R: ml468, full, L: ml469, full, y: ml472, full, P11: ml474, ipiv, tmp28: ml473, full
    ml475 = [1:length(ml474);]
    @inbounds for i in 1:length(ml474)
        ml475[i], ml475[ml474[i]] = ml475[ml474[i]], ml475[i];
    end;
    ml476 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml476 = ml473[invperm(ml475),:]

    # R: ml468, full, L: ml469, full, y: ml472, full, tmp25: ml476, full
    ml477 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml476, 0.0, ml477)

    # R: ml468, full, L: ml469, full, y: ml472, full, tmp19: ml477, symmetric_lower_triangular
    ml478 = diag(ml469)
    ml479 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml468, 1, ml479, 1)
    # tmp29 = (L R)
    for i = 1:size(ml468, 2);
        view(ml468, :, i)[:] .*= ml478;
    end;        

    # R: ml479, full, y: ml472, full, tmp19: ml477, symmetric_lower_triangular, tmp29: ml468, full
    for i = 1:2000-1;
        view(ml477, i, i+1:2000)[:] = view(ml477, i+1:2000, i);
    end;
    ml480 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml477, 1, ml480, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml468, ml479, 1.0, ml477)

    # y: ml472, full, tmp19: ml480, full, tmp31: ml477, full
    ml481 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml477, ml481, info) = LinearAlgebra.LAPACK.getrf!(ml477)

    # y: ml472, full, tmp19: ml480, full, P35: ml481, ipiv, L33: ml477, lower_triangular_udiag, U34: ml477, upper_triangular
    ml482 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml480, ml472, 0.0, ml482)

    # P35: ml481, ipiv, L33: ml477, lower_triangular_udiag, U34: ml477, upper_triangular, tmp32: ml482, full
    ml483 = [1:length(ml481);]
    @inbounds for i in 1:length(ml481)
        ml483[i], ml483[ml481[i]] = ml483[ml481[i]], ml483[i];
    end;
    ml484 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml484 = ml482[ml483]

    # L33: ml477, lower_triangular_udiag, U34: ml477, upper_triangular, tmp40: ml484, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml477, ml484)

    # U34: ml477, upper_triangular, tmp41: ml484, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml477, ml484)

    # tmp17: ml484, full
    # x = tmp17
    return (ml484)
end