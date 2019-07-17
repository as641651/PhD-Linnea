using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm31(ml1036::Array{Float64,2}, ml1037::Array{Float64,2}, ml1038::Array{Float64,2}, ml1039::Array{Float64,2}, ml1040::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1036, full, L: ml1037, full, A: ml1038, full, B: ml1039, full, y: ml1040, full
    ml1041 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1041, ml1039)

    # R: ml1036, full, L: ml1037, full, A: ml1038, full, y: ml1040, full, tmp26: ml1041, full
    ml1042 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1038, ml1042, info) = LinearAlgebra.LAPACK.getrf!(ml1038)

    # R: ml1036, full, L: ml1037, full, y: ml1040, full, tmp26: ml1041, full, P11: ml1042, ipiv, L9: ml1038, lower_triangular_udiag, U10: ml1038, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1038, ml1041)

    # R: ml1036, full, L: ml1037, full, y: ml1040, full, P11: ml1042, ipiv, L9: ml1038, lower_triangular_udiag, tmp27: ml1041, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1038, ml1041)

    # R: ml1036, full, L: ml1037, full, y: ml1040, full, P11: ml1042, ipiv, tmp28: ml1041, full
    ml1043 = [1:length(ml1042);]
    @inbounds for i in 1:length(ml1042)
        ml1043[i], ml1043[ml1042[i]] = ml1043[ml1042[i]], ml1043[i];
    end;
    ml1044 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1044 = ml1041[invperm(ml1043),:]

    # R: ml1036, full, L: ml1037, full, y: ml1040, full, tmp25: ml1044, full
    ml1045 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1044, 0.0, ml1045)

    # R: ml1036, full, L: ml1037, full, y: ml1040, full, tmp19: ml1045, symmetric_lower_triangular
    ml1046 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1045, ml1040, 0.0, ml1046)

    # R: ml1036, full, L: ml1037, full, tmp19: ml1045, symmetric_lower_triangular, tmp32: ml1046, full
    ml1047 = diag(ml1037)
    ml1048 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1036, 1, ml1048, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1036, 2);
        view(ml1036, :, i)[:] .*= ml1047;
    end;        

    # R: ml1048, full, tmp19: ml1045, symmetric_lower_triangular, tmp32: ml1046, full, tmp29: ml1036, full
    for i = 1:2000-1;
        view(ml1045, i, i+1:2000)[:] = view(ml1045, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml1048, ml1036, 1.0, ml1045)

    # tmp32: ml1046, full, tmp31: ml1045, full
    ml1049 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1045, ml1049, info) = LinearAlgebra.LAPACK.getrf!(ml1045)

    # tmp32: ml1046, full, P35: ml1049, ipiv, L33: ml1045, lower_triangular_udiag, U34: ml1045, upper_triangular
    ml1050 = [1:length(ml1049);]
    @inbounds for i in 1:length(ml1049)
        ml1050[i], ml1050[ml1049[i]] = ml1050[ml1049[i]], ml1050[i];
    end;
    ml1051 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1051 = ml1046[ml1050]

    # L33: ml1045, lower_triangular_udiag, U34: ml1045, upper_triangular, tmp40: ml1051, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1045, ml1051)

    # U34: ml1045, upper_triangular, tmp41: ml1051, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1045, ml1051)

    # tmp17: ml1051, full
    # x = tmp17
    return (ml1051)
end