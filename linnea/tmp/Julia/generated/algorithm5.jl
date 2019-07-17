using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm5(ml162::Array{Float64,2}, ml163::Array{Float64,2}, ml164::Array{Float64,2}, ml165::Array{Float64,2}, ml166::Array{Float64,1})
    # cost 5.07e+10
    # R: ml162, full, L: ml163, full, A: ml164, full, B: ml165, full, y: ml166, full
    ml167 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml164, ml167, info) = LinearAlgebra.LAPACK.getrf!(ml164)

    # R: ml162, full, L: ml163, full, B: ml165, full, y: ml166, full, P11: ml167, ipiv, L9: ml164, lower_triangular_udiag, U10: ml164, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml164, ml165)

    # R: ml162, full, L: ml163, full, y: ml166, full, P11: ml167, ipiv, L9: ml164, lower_triangular_udiag, tmp53: ml165, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml164, ml165)

    # R: ml162, full, L: ml163, full, y: ml166, full, P11: ml167, ipiv, tmp54: ml165, full
    ml168 = [1:length(ml167);]
    @inbounds for i in 1:length(ml167)
        ml168[i], ml168[ml167[i]] = ml168[ml167[i]], ml168[i];
    end;
    ml169 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml169 = ml165[:,invperm(ml168)]

    # R: ml162, full, L: ml163, full, y: ml166, full, tmp55: ml169, full
    ml170 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml170, ml169)

    # R: ml162, full, L: ml163, full, y: ml166, full, tmp25: ml170, full
    ml171 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml170, 0.0, ml171)

    # R: ml162, full, L: ml163, full, y: ml166, full, tmp19: ml171, symmetric_lower_triangular
    ml172 = diag(ml163)
    ml173 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml162, 1, ml173, 1)
    # tmp29 = (L R)
    for i = 1:size(ml162, 2);
        view(ml162, :, i)[:] .*= ml172;
    end;        

    # R: ml173, full, y: ml166, full, tmp19: ml171, symmetric_lower_triangular, tmp29: ml162, full
    for i = 1:2000-1;
        view(ml171, i, i+1:2000)[:] = view(ml171, i+1:2000, i);
    end;
    ml174 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml171, 1, ml174, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml162, ml173, 1.0, ml171)

    # y: ml166, full, tmp19: ml174, full, tmp31: ml171, full
    ml175 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml171, ml175, info) = LinearAlgebra.LAPACK.getrf!(ml171)

    # y: ml166, full, tmp19: ml174, full, P35: ml175, ipiv, L33: ml171, lower_triangular_udiag, U34: ml171, upper_triangular
    ml176 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml174, ml166, 0.0, ml176)

    # P35: ml175, ipiv, L33: ml171, lower_triangular_udiag, U34: ml171, upper_triangular, tmp32: ml176, full
    ml177 = [1:length(ml175);]
    @inbounds for i in 1:length(ml175)
        ml177[i], ml177[ml175[i]] = ml177[ml175[i]], ml177[i];
    end;
    ml178 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml178 = ml176[ml177]

    # L33: ml171, lower_triangular_udiag, U34: ml171, upper_triangular, tmp40: ml178, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml171, ml178)

    # U34: ml171, upper_triangular, tmp41: ml178, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml171, ml178)

    # tmp17: ml178, full
    # x = tmp17
    return (ml178)
end