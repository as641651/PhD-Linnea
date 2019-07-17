using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm29(ml970::Array{Float64,2}, ml971::Array{Float64,2}, ml972::Array{Float64,2}, ml973::Array{Float64,2}, ml974::Array{Float64,1})
    # cost 5.07e+10
    # R: ml970, full, L: ml971, full, A: ml972, full, B: ml973, full, y: ml974, full
    ml975 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml972, ml975, info) = LinearAlgebra.LAPACK.getrf!(ml972)

    # R: ml970, full, L: ml971, full, B: ml973, full, y: ml974, full, P11: ml975, ipiv, L9: ml972, lower_triangular_udiag, U10: ml972, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml972, ml973)

    # R: ml970, full, L: ml971, full, y: ml974, full, P11: ml975, ipiv, L9: ml972, lower_triangular_udiag, tmp53: ml973, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml972, ml973)

    # R: ml970, full, L: ml971, full, y: ml974, full, P11: ml975, ipiv, tmp54: ml973, full
    ml976 = [1:length(ml975);]
    @inbounds for i in 1:length(ml975)
        ml976[i], ml976[ml975[i]] = ml976[ml975[i]], ml976[i];
    end;
    ml977 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml977 = ml973[:,invperm(ml976)]

    # R: ml970, full, L: ml971, full, y: ml974, full, tmp55: ml977, full
    ml978 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml978, ml977)

    # R: ml970, full, L: ml971, full, y: ml974, full, tmp25: ml978, full
    ml979 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml978, 0.0, ml979)

    # R: ml970, full, L: ml971, full, y: ml974, full, tmp19: ml979, symmetric_lower_triangular
    ml980 = diag(ml971)
    ml981 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml970, 1, ml981, 1)
    # tmp29 = (L R)
    for i = 1:size(ml970, 2);
        view(ml970, :, i)[:] .*= ml980;
    end;        

    # R: ml981, full, y: ml974, full, tmp19: ml979, symmetric_lower_triangular, tmp29: ml970, full
    for i = 1:2000-1;
        view(ml979, i, i+1:2000)[:] = view(ml979, i+1:2000, i);
    end;
    ml982 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml979, 1, ml982, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml970, ml981, 1.0, ml979)

    # y: ml974, full, tmp19: ml982, full, tmp31: ml979, full
    ml983 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml979, ml983, info) = LinearAlgebra.LAPACK.getrf!(ml979)

    # y: ml974, full, tmp19: ml982, full, P35: ml983, ipiv, L33: ml979, lower_triangular_udiag, U34: ml979, upper_triangular
    ml984 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml982, ml974, 0.0, ml984)

    # P35: ml983, ipiv, L33: ml979, lower_triangular_udiag, U34: ml979, upper_triangular, tmp32: ml984, full
    ml985 = [1:length(ml983);]
    @inbounds for i in 1:length(ml983)
        ml985[i], ml985[ml983[i]] = ml985[ml983[i]], ml985[i];
    end;
    ml986 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml986 = ml984[ml985]

    # L33: ml979, lower_triangular_udiag, U34: ml979, upper_triangular, tmp40: ml986, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml979, ml986)

    # U34: ml979, upper_triangular, tmp41: ml986, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml979, ml986)

    # tmp17: ml986, full
    # x = tmp17
    return (ml986)
end