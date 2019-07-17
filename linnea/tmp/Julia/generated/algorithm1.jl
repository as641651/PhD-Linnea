using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm1(ml32::Array{Float64,2}, ml33::Array{Float64,2}, ml34::Array{Float64,2}, ml35::Array{Float64,2}, ml36::Array{Float64,1})
    # cost 5.07e+10
    # R: ml32, full, L: ml33, full, A: ml34, full, B: ml35, full, y: ml36, full
    ml37 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml34, ml37, info) = LinearAlgebra.LAPACK.getrf!(ml34)

    # R: ml32, full, L: ml33, full, B: ml35, full, y: ml36, full, P11: ml37, ipiv, L9: ml34, lower_triangular_udiag, U10: ml34, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml34, ml35)

    # R: ml32, full, L: ml33, full, y: ml36, full, P11: ml37, ipiv, L9: ml34, lower_triangular_udiag, tmp53: ml35, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml34, ml35)

    # R: ml32, full, L: ml33, full, y: ml36, full, P11: ml37, ipiv, tmp54: ml35, full
    ml38 = [1:length(ml37);]
    @inbounds for i in 1:length(ml37)
        ml38[i], ml38[ml37[i]] = ml38[ml37[i]], ml38[i];
    end;
    ml39 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml39 = ml35[:,invperm(ml38)]

    # R: ml32, full, L: ml33, full, y: ml36, full, tmp55: ml39, full
    ml40 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml40, ml39)

    # R: ml32, full, L: ml33, full, y: ml36, full, tmp25: ml40, full
    ml41 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml40, 0.0, ml41)

    # R: ml32, full, L: ml33, full, y: ml36, full, tmp19: ml41, symmetric_lower_triangular
    ml42 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml41, ml36, 0.0, ml42)

    # R: ml32, full, L: ml33, full, tmp19: ml41, symmetric_lower_triangular, tmp32: ml42, full
    ml43 = diag(ml33)
    ml44 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml32, 1, ml44, 1)
    # tmp29 = (L R)
    for i = 1:size(ml32, 2);
        view(ml32, :, i)[:] .*= ml43;
    end;        

    # R: ml44, full, tmp19: ml41, symmetric_lower_triangular, tmp32: ml42, full, tmp29: ml32, full
    for i = 1:2000-1;
        view(ml41, i, i+1:2000)[:] = view(ml41, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml44, ml32, 1.0, ml41)

    # tmp32: ml42, full, tmp31: ml41, full
    ml45 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml41, ml45, info) = LinearAlgebra.LAPACK.getrf!(ml41)

    # tmp32: ml42, full, P35: ml45, ipiv, L33: ml41, lower_triangular_udiag, U34: ml41, upper_triangular
    ml46 = [1:length(ml45);]
    @inbounds for i in 1:length(ml45)
        ml46[i], ml46[ml45[i]] = ml46[ml45[i]], ml46[i];
    end;
    ml47 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml47 = ml42[ml46]

    # L33: ml41, lower_triangular_udiag, U34: ml41, upper_triangular, tmp40: ml47, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml41, ml47)

    # U34: ml41, upper_triangular, tmp41: ml47, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml41, ml47)

    # tmp17: ml47, full
    # x = tmp17
    return (ml47)
end