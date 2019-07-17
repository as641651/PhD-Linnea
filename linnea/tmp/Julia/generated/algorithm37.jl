using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm37(ml1234::Array{Float64,2}, ml1235::Array{Float64,2}, ml1236::Array{Float64,2}, ml1237::Array{Float64,2}, ml1238::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1234, full, L: ml1235, full, A: ml1236, full, B: ml1237, full, y: ml1238, full
    ml1239 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1239, ml1237)

    # R: ml1234, full, L: ml1235, full, A: ml1236, full, y: ml1238, full, tmp26: ml1239, full
    ml1240 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1236, ml1240, info) = LinearAlgebra.LAPACK.getrf!(ml1236)

    # R: ml1234, full, L: ml1235, full, y: ml1238, full, tmp26: ml1239, full, P11: ml1240, ipiv, L9: ml1236, lower_triangular_udiag, U10: ml1236, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1236, ml1239)

    # R: ml1234, full, L: ml1235, full, y: ml1238, full, P11: ml1240, ipiv, L9: ml1236, lower_triangular_udiag, tmp27: ml1239, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1236, ml1239)

    # R: ml1234, full, L: ml1235, full, y: ml1238, full, P11: ml1240, ipiv, tmp28: ml1239, full
    ml1241 = [1:length(ml1240);]
    @inbounds for i in 1:length(ml1240)
        ml1241[i], ml1241[ml1240[i]] = ml1241[ml1240[i]], ml1241[i];
    end;
    ml1242 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1242 = ml1239[invperm(ml1241),:]

    # R: ml1234, full, L: ml1235, full, y: ml1238, full, tmp25: ml1242, full
    ml1243 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1242, 0.0, ml1243)

    # R: ml1234, full, L: ml1235, full, y: ml1238, full, tmp19: ml1243, symmetric_lower_triangular
    ml1244 = diag(ml1235)
    ml1245 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1234, 1, ml1245, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1234, 2);
        view(ml1234, :, i)[:] .*= ml1244;
    end;        

    # R: ml1245, full, y: ml1238, full, tmp19: ml1243, symmetric_lower_triangular, tmp29: ml1234, full
    for i = 1:2000-1;
        view(ml1243, i, i+1:2000)[:] = view(ml1243, i+1:2000, i);
    end;
    ml1246 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1243, 1, ml1246, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1234, ml1245, 1.0, ml1243)

    # y: ml1238, full, tmp19: ml1246, full, tmp31: ml1243, full
    ml1247 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1243, ml1247, info) = LinearAlgebra.LAPACK.getrf!(ml1243)

    # y: ml1238, full, tmp19: ml1246, full, P35: ml1247, ipiv, L33: ml1243, lower_triangular_udiag, U34: ml1243, upper_triangular
    ml1248 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1246, ml1238, 0.0, ml1248)

    # P35: ml1247, ipiv, L33: ml1243, lower_triangular_udiag, U34: ml1243, upper_triangular, tmp32: ml1248, full
    ml1249 = [1:length(ml1247);]
    @inbounds for i in 1:length(ml1247)
        ml1249[i], ml1249[ml1247[i]] = ml1249[ml1247[i]], ml1249[i];
    end;
    ml1250 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1250 = ml1248[ml1249]

    # L33: ml1243, lower_triangular_udiag, U34: ml1243, upper_triangular, tmp40: ml1250, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1243, ml1250)

    # U34: ml1243, upper_triangular, tmp41: ml1250, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1243, ml1250)

    # tmp17: ml1250, full
    # x = tmp17
    return (ml1250)
end