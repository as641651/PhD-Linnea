using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm26(ml868::Array{Float64,2}, ml869::Array{Float64,2}, ml870::Array{Float64,2}, ml871::Array{Float64,2}, ml872::Array{Float64,1})
    # cost 5.07e+10
    # R: ml868, full, L: ml869, full, A: ml870, full, B: ml871, full, y: ml872, full
    ml873 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml873, ml871)

    # R: ml868, full, L: ml869, full, A: ml870, full, y: ml872, full, tmp26: ml873, full
    ml874 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml870, ml874, info) = LinearAlgebra.LAPACK.getrf!(ml870)

    # R: ml868, full, L: ml869, full, y: ml872, full, tmp26: ml873, full, P11: ml874, ipiv, L9: ml870, lower_triangular_udiag, U10: ml870, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml870, ml873)

    # R: ml868, full, L: ml869, full, y: ml872, full, P11: ml874, ipiv, L9: ml870, lower_triangular_udiag, tmp27: ml873, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml870, ml873)

    # R: ml868, full, L: ml869, full, y: ml872, full, P11: ml874, ipiv, tmp28: ml873, full
    ml875 = [1:length(ml874);]
    @inbounds for i in 1:length(ml874)
        ml875[i], ml875[ml874[i]] = ml875[ml874[i]], ml875[i];
    end;
    ml876 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml876 = ml873[invperm(ml875),:]

    # R: ml868, full, L: ml869, full, y: ml872, full, tmp25: ml876, full
    ml877 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml876, 0.0, ml877)

    # R: ml868, full, L: ml869, full, y: ml872, full, tmp19: ml877, symmetric_lower_triangular
    ml878 = diag(ml869)
    ml879 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml868, 1, ml879, 1)
    # tmp29 = (L R)
    for i = 1:size(ml868, 2);
        view(ml868, :, i)[:] .*= ml878;
    end;        

    # R: ml879, full, y: ml872, full, tmp19: ml877, symmetric_lower_triangular, tmp29: ml868, full
    for i = 1:2000-1;
        view(ml877, i, i+1:2000)[:] = view(ml877, i+1:2000, i);
    end;
    ml880 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml877, 1, ml880, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml868, ml879, 1.0, ml877)

    # y: ml872, full, tmp19: ml880, full, tmp31: ml877, full
    ml881 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml877, ml881, info) = LinearAlgebra.LAPACK.getrf!(ml877)

    # y: ml872, full, tmp19: ml880, full, P35: ml881, ipiv, L33: ml877, lower_triangular_udiag, U34: ml877, upper_triangular
    ml882 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml880, ml872, 0.0, ml882)

    # P35: ml881, ipiv, L33: ml877, lower_triangular_udiag, U34: ml877, upper_triangular, tmp32: ml882, full
    ml883 = [1:length(ml881);]
    @inbounds for i in 1:length(ml881)
        ml883[i], ml883[ml881[i]] = ml883[ml881[i]], ml883[i];
    end;
    ml884 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml884 = ml882[ml883]

    # L33: ml877, lower_triangular_udiag, U34: ml877, upper_triangular, tmp40: ml884, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml877, ml884)

    # U34: ml877, upper_triangular, tmp41: ml884, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml877, ml884)

    # tmp17: ml884, full
    # x = tmp17
    return (ml884)
end