using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm3(ml96::Array{Float64,2}, ml97::Array{Float64,2}, ml98::Array{Float64,2}, ml99::Array{Float64,2}, ml100::Array{Float64,1})
    # cost 5.07e+10
    # R: ml96, full, L: ml97, full, A: ml98, full, B: ml99, full, y: ml100, full
    ml101 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml98, ml101, info) = LinearAlgebra.LAPACK.getrf!(ml98)

    # R: ml96, full, L: ml97, full, B: ml99, full, y: ml100, full, P11: ml101, ipiv, L9: ml98, lower_triangular_udiag, U10: ml98, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml98, ml99)

    # R: ml96, full, L: ml97, full, y: ml100, full, P11: ml101, ipiv, L9: ml98, lower_triangular_udiag, tmp53: ml99, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml98, ml99)

    # R: ml96, full, L: ml97, full, y: ml100, full, P11: ml101, ipiv, tmp54: ml99, full
    ml102 = [1:length(ml101);]
    @inbounds for i in 1:length(ml101)
        ml102[i], ml102[ml101[i]] = ml102[ml101[i]], ml102[i];
    end;
    ml103 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml103 = ml99[:,invperm(ml102)]

    # R: ml96, full, L: ml97, full, y: ml100, full, tmp55: ml103, full
    ml104 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml104, ml103)

    # R: ml96, full, L: ml97, full, y: ml100, full, tmp25: ml104, full
    ml105 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml104, 0.0, ml105)

    # R: ml96, full, L: ml97, full, y: ml100, full, tmp19: ml105, symmetric_lower_triangular
    ml106 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml105, ml100, 0.0, ml106)

    # R: ml96, full, L: ml97, full, tmp19: ml105, symmetric_lower_triangular, tmp32: ml106, full
    ml107 = diag(ml97)
    ml108 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml96, 1, ml108, 1)
    # tmp29 = (L R)
    for i = 1:size(ml96, 2);
        view(ml96, :, i)[:] .*= ml107;
    end;        

    # R: ml108, full, tmp19: ml105, symmetric_lower_triangular, tmp32: ml106, full, tmp29: ml96, full
    for i = 1:2000-1;
        view(ml105, i, i+1:2000)[:] = view(ml105, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml108, ml96, 1.0, ml105)

    # tmp32: ml106, full, tmp31: ml105, full
    ml109 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml105, ml109, info) = LinearAlgebra.LAPACK.getrf!(ml105)

    # tmp32: ml106, full, P35: ml109, ipiv, L33: ml105, lower_triangular_udiag, U34: ml105, upper_triangular
    ml110 = [1:length(ml109);]
    @inbounds for i in 1:length(ml109)
        ml110[i], ml110[ml109[i]] = ml110[ml109[i]], ml110[i];
    end;
    ml111 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml111 = ml106[ml110]

    # L33: ml105, lower_triangular_udiag, U34: ml105, upper_triangular, tmp40: ml111, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml105, ml111)

    # U34: ml105, upper_triangular, tmp41: ml111, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml105, ml111)

    # tmp17: ml111, full
    # x = tmp17
    return (ml111)
end