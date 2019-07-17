using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm22(ml736::Array{Float64,2}, ml737::Array{Float64,2}, ml738::Array{Float64,2}, ml739::Array{Float64,2}, ml740::Array{Float64,1})
    # cost 5.07e+10
    # R: ml736, full, L: ml737, full, A: ml738, full, B: ml739, full, y: ml740, full
    ml741 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml738, ml741, info) = LinearAlgebra.LAPACK.getrf!(ml738)

    # R: ml736, full, L: ml737, full, B: ml739, full, y: ml740, full, P11: ml741, ipiv, L9: ml738, lower_triangular_udiag, U10: ml738, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml738, ml739)

    # R: ml736, full, L: ml737, full, y: ml740, full, P11: ml741, ipiv, L9: ml738, lower_triangular_udiag, tmp53: ml739, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml738, ml739)

    # R: ml736, full, L: ml737, full, y: ml740, full, P11: ml741, ipiv, tmp54: ml739, full
    ml742 = [1:length(ml741);]
    @inbounds for i in 1:length(ml741)
        ml742[i], ml742[ml741[i]] = ml742[ml741[i]], ml742[i];
    end;
    ml743 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml743 = ml739[:,invperm(ml742)]

    # R: ml736, full, L: ml737, full, y: ml740, full, tmp55: ml743, full
    ml744 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml744, ml743)

    # R: ml736, full, L: ml737, full, y: ml740, full, tmp25: ml744, full
    ml745 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml744, 0.0, ml745)

    # R: ml736, full, L: ml737, full, y: ml740, full, tmp19: ml745, symmetric_lower_triangular
    ml746 = diag(ml737)
    ml747 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml736, 1, ml747, 1)
    # tmp29 = (L R)
    for i = 1:size(ml736, 2);
        view(ml736, :, i)[:] .*= ml746;
    end;        

    # R: ml747, full, y: ml740, full, tmp19: ml745, symmetric_lower_triangular, tmp29: ml736, full
    ml748 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml745, ml740, 0.0, ml748)

    # R: ml747, full, tmp19: ml745, symmetric_lower_triangular, tmp29: ml736, full, tmp32: ml748, full
    for i = 1:2000-1;
        view(ml745, i, i+1:2000)[:] = view(ml745, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml736, ml747, 1.0, ml745)

    # tmp32: ml748, full, tmp31: ml745, full
    ml749 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml745, ml749, info) = LinearAlgebra.LAPACK.getrf!(ml745)

    # tmp32: ml748, full, P35: ml749, ipiv, L33: ml745, lower_triangular_udiag, U34: ml745, upper_triangular
    ml750 = [1:length(ml749);]
    @inbounds for i in 1:length(ml749)
        ml750[i], ml750[ml749[i]] = ml750[ml749[i]], ml750[i];
    end;
    ml751 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml751 = ml748[ml750]

    # L33: ml745, lower_triangular_udiag, U34: ml745, upper_triangular, tmp40: ml751, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml745, ml751)

    # U34: ml745, upper_triangular, tmp41: ml751, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml745, ml751)

    # tmp17: ml751, full
    # x = tmp17
    return (ml751)
end