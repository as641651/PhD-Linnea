using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm30(ml1004::Array{Float64,2}, ml1005::Array{Float64,2}, ml1006::Array{Float64,2}, ml1007::Array{Float64,2}, ml1008::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1004, full, L: ml1005, full, A: ml1006, full, B: ml1007, full, y: ml1008, full
    ml1009 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1009, ml1007)

    # R: ml1004, full, L: ml1005, full, A: ml1006, full, y: ml1008, full, tmp26: ml1009, full
    ml1010 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1006, ml1010, info) = LinearAlgebra.LAPACK.getrf!(ml1006)

    # R: ml1004, full, L: ml1005, full, y: ml1008, full, tmp26: ml1009, full, P11: ml1010, ipiv, L9: ml1006, lower_triangular_udiag, U10: ml1006, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1006, ml1009)

    # R: ml1004, full, L: ml1005, full, y: ml1008, full, P11: ml1010, ipiv, L9: ml1006, lower_triangular_udiag, tmp27: ml1009, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1006, ml1009)

    # R: ml1004, full, L: ml1005, full, y: ml1008, full, P11: ml1010, ipiv, tmp28: ml1009, full
    ml1011 = [1:length(ml1010);]
    @inbounds for i in 1:length(ml1010)
        ml1011[i], ml1011[ml1010[i]] = ml1011[ml1010[i]], ml1011[i];
    end;
    ml1012 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1012 = ml1009[invperm(ml1011),:]

    # R: ml1004, full, L: ml1005, full, y: ml1008, full, tmp25: ml1012, full
    ml1013 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1012, 0.0, ml1013)

    # R: ml1004, full, L: ml1005, full, y: ml1008, full, tmp19: ml1013, symmetric_lower_triangular
    ml1014 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1013, ml1008, 0.0, ml1014)

    # R: ml1004, full, L: ml1005, full, tmp19: ml1013, symmetric_lower_triangular, tmp32: ml1014, full
    ml1015 = diag(ml1005)
    ml1016 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1004, 1, ml1016, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1004, 2);
        view(ml1004, :, i)[:] .*= ml1015;
    end;        

    # R: ml1016, full, tmp19: ml1013, symmetric_lower_triangular, tmp32: ml1014, full, tmp29: ml1004, full
    for i = 1:2000-1;
        view(ml1013, i, i+1:2000)[:] = view(ml1013, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1016, ml1004, 1.0, ml1013)

    # tmp32: ml1014, full, tmp31: ml1013, full
    ml1017 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1013, ml1017, info) = LinearAlgebra.LAPACK.getrf!(ml1013)

    # tmp32: ml1014, full, P35: ml1017, ipiv, L33: ml1013, lower_triangular_udiag, U34: ml1013, upper_triangular
    ml1018 = [1:length(ml1017);]
    @inbounds for i in 1:length(ml1017)
        ml1018[i], ml1018[ml1017[i]] = ml1018[ml1017[i]], ml1018[i];
    end;
    ml1019 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1019 = ml1014[ml1018]

    # L33: ml1013, lower_triangular_udiag, U34: ml1013, upper_triangular, tmp40: ml1019, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1013, ml1019)

    # U34: ml1013, upper_triangular, tmp41: ml1019, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1013, ml1019)

    # tmp17: ml1019, full
    # x = tmp17
    return (ml1019)
end