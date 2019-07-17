using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm0(ml0::Array{Float64,2}, ml1::Array{Float64,2}, ml2::Array{Float64,2}, ml3::Array{Float64,2}, ml4::Array{Float64,1})
    # cost 5.07e+10
    # R: ml0, full, L: ml1, full, A: ml2, full, B: ml3, full, y: ml4, full
    ml5 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2, ml5, info) = LinearAlgebra.LAPACK.getrf!(ml2)

    # R: ml0, full, L: ml1, full, B: ml3, full, y: ml4, full, P11: ml5, ipiv, L9: ml2, lower_triangular_udiag, U10: ml2, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml2, ml3)

    # R: ml0, full, L: ml1, full, y: ml4, full, P11: ml5, ipiv, L9: ml2, lower_triangular_udiag, tmp53: ml3, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml2, ml3)

    # R: ml0, full, L: ml1, full, y: ml4, full, P11: ml5, ipiv, tmp54: ml3, full
    ml6 = [1:length(ml5);]
    @inbounds for i in 1:length(ml5)
        ml6[i], ml6[ml5[i]] = ml6[ml5[i]], ml6[i];
    end;
    ml7 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml7 = ml3[:,invperm(ml6)]

    # R: ml0, full, L: ml1, full, y: ml4, full, tmp55: ml7, full
    ml8 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml8, ml7)

    # R: ml0, full, L: ml1, full, y: ml4, full, tmp25: ml8, full
    ml9 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml8, 0.0, ml9)

    # R: ml0, full, L: ml1, full, y: ml4, full, tmp19: ml9, symmetric_lower_triangular
    ml10 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml9, ml4, 0.0, ml10)

    # R: ml0, full, L: ml1, full, tmp19: ml9, symmetric_lower_triangular, tmp32: ml10, full
    ml11 = diag(ml1)
    ml12 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml0, 1, ml12, 1)
    # tmp29 = (L R)
    for i = 1:size(ml0, 2);
        view(ml0, :, i)[:] .*= ml11;
    end;        

    # R: ml12, full, tmp19: ml9, symmetric_lower_triangular, tmp32: ml10, full, tmp29: ml0, full
    for i = 1:2000-1;
        view(ml9, i, i+1:2000)[:] = view(ml9, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml12, ml0, 1.0, ml9)

    # tmp32: ml10, full, tmp31: ml9, full
    ml13 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml9, ml13, info) = LinearAlgebra.LAPACK.getrf!(ml9)

    # tmp32: ml10, full, P35: ml13, ipiv, L33: ml9, lower_triangular_udiag, U34: ml9, upper_triangular
    ml14 = [1:length(ml13);]
    @inbounds for i in 1:length(ml13)
        ml14[i], ml14[ml13[i]] = ml14[ml13[i]], ml14[i];
    end;
    ml15 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml15 = ml10[ml14]

    # L33: ml9, lower_triangular_udiag, U34: ml9, upper_triangular, tmp40: ml15, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml9, ml15)

    # U34: ml9, upper_triangular, tmp41: ml15, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml9, ml15)

    # tmp17: ml15, full
    # x = tmp17
    return (ml15)
end