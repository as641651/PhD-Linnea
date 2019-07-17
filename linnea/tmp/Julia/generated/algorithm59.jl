using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm59(ml1974::Array{Float64,2}, ml1975::Array{Float64,2}, ml1976::Array{Float64,2}, ml1977::Array{Float64,2}, ml1978::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1974, full, L: ml1975, full, A: ml1976, full, B: ml1977, full, y: ml1978, full
    ml1979 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1976, ml1979, info) = LinearAlgebra.LAPACK.getrf!(ml1976)

    # R: ml1974, full, L: ml1975, full, B: ml1977, full, y: ml1978, full, P11: ml1979, ipiv, L9: ml1976, lower_triangular_udiag, U10: ml1976, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1976, ml1977)

    # R: ml1974, full, L: ml1975, full, y: ml1978, full, P11: ml1979, ipiv, L9: ml1976, lower_triangular_udiag, tmp53: ml1977, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1976, ml1977)

    # R: ml1974, full, L: ml1975, full, y: ml1978, full, P11: ml1979, ipiv, tmp54: ml1977, full
    ml1980 = [1:length(ml1979);]
    @inbounds for i in 1:length(ml1979)
        ml1980[i], ml1980[ml1979[i]] = ml1980[ml1979[i]], ml1980[i];
    end;
    ml1981 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1981 = ml1977[:,invperm(ml1980)]

    # R: ml1974, full, L: ml1975, full, y: ml1978, full, tmp55: ml1981, full
    ml1982 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1982, ml1981)

    # R: ml1974, full, L: ml1975, full, y: ml1978, full, tmp25: ml1982, full
    ml1983 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1982, 0.0, ml1983)

    # R: ml1974, full, L: ml1975, full, y: ml1978, full, tmp19: ml1983, symmetric_lower_triangular
    ml1984 = diag(ml1975)
    ml1985 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1974, 1, ml1985, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1974, 2);
        view(ml1974, :, i)[:] .*= ml1984;
    end;        

    # R: ml1985, full, y: ml1978, full, tmp19: ml1983, symmetric_lower_triangular, tmp29: ml1974, full
    for i = 1:2000-1;
        view(ml1983, i, i+1:2000)[:] = view(ml1983, i+1:2000, i);
    end;
    ml1986 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1983, 1, ml1986, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1974, ml1985, 1.0, ml1983)

    # y: ml1978, full, tmp19: ml1986, full, tmp31: ml1983, full
    ml1987 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1983, ml1987, info) = LinearAlgebra.LAPACK.getrf!(ml1983)

    # y: ml1978, full, tmp19: ml1986, full, P35: ml1987, ipiv, L33: ml1983, lower_triangular_udiag, U34: ml1983, upper_triangular
    ml1988 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1986, ml1978, 0.0, ml1988)

    # P35: ml1987, ipiv, L33: ml1983, lower_triangular_udiag, U34: ml1983, upper_triangular, tmp32: ml1988, full
    ml1989 = [1:length(ml1987);]
    @inbounds for i in 1:length(ml1987)
        ml1989[i], ml1989[ml1987[i]] = ml1989[ml1987[i]], ml1989[i];
    end;
    ml1990 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1990 = ml1988[ml1989]

    # L33: ml1983, lower_triangular_udiag, U34: ml1983, upper_triangular, tmp40: ml1990, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1983, ml1990)

    # U34: ml1983, upper_triangular, tmp41: ml1990, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1983, ml1990)

    # tmp17: ml1990, full
    # x = tmp17
    return (ml1990)
end