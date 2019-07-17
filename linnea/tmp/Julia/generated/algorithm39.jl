using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm39(ml1302::Array{Float64,2}, ml1303::Array{Float64,2}, ml1304::Array{Float64,2}, ml1305::Array{Float64,2}, ml1306::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1302, full, L: ml1303, full, A: ml1304, full, B: ml1305, full, y: ml1306, full
    ml1307 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1307, ml1305)

    # R: ml1302, full, L: ml1303, full, A: ml1304, full, y: ml1306, full, tmp26: ml1307, full
    ml1308 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1304, ml1308, info) = LinearAlgebra.LAPACK.getrf!(ml1304)

    # R: ml1302, full, L: ml1303, full, y: ml1306, full, tmp26: ml1307, full, P11: ml1308, ipiv, L9: ml1304, lower_triangular_udiag, U10: ml1304, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1304, ml1307)

    # R: ml1302, full, L: ml1303, full, y: ml1306, full, P11: ml1308, ipiv, L9: ml1304, lower_triangular_udiag, tmp27: ml1307, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1304, ml1307)

    # R: ml1302, full, L: ml1303, full, y: ml1306, full, P11: ml1308, ipiv, tmp28: ml1307, full
    ml1309 = [1:length(ml1308);]
    @inbounds for i in 1:length(ml1308)
        ml1309[i], ml1309[ml1308[i]] = ml1309[ml1308[i]], ml1309[i];
    end;
    ml1310 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1310 = ml1307[invperm(ml1309),:]

    # R: ml1302, full, L: ml1303, full, y: ml1306, full, tmp25: ml1310, full
    ml1311 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1310, 0.0, ml1311)

    # R: ml1302, full, L: ml1303, full, y: ml1306, full, tmp19: ml1311, symmetric_lower_triangular
    ml1312 = diag(ml1303)
    ml1313 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1302, 1, ml1313, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1302, 2);
        view(ml1302, :, i)[:] .*= ml1312;
    end;        

    # R: ml1313, full, y: ml1306, full, tmp19: ml1311, symmetric_lower_triangular, tmp29: ml1302, full
    for i = 1:2000-1;
        view(ml1311, i, i+1:2000)[:] = view(ml1311, i+1:2000, i);
    end;
    ml1314 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1311, 1, ml1314, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1302, ml1313, 1.0, ml1311)

    # y: ml1306, full, tmp19: ml1314, full, tmp31: ml1311, full
    ml1315 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1311, ml1315, info) = LinearAlgebra.LAPACK.getrf!(ml1311)

    # y: ml1306, full, tmp19: ml1314, full, P35: ml1315, ipiv, L33: ml1311, lower_triangular_udiag, U34: ml1311, upper_triangular
    ml1316 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1314, ml1306, 0.0, ml1316)

    # P35: ml1315, ipiv, L33: ml1311, lower_triangular_udiag, U34: ml1311, upper_triangular, tmp32: ml1316, full
    ml1317 = [1:length(ml1315);]
    @inbounds for i in 1:length(ml1315)
        ml1317[i], ml1317[ml1315[i]] = ml1317[ml1315[i]], ml1317[i];
    end;
    ml1318 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1318 = ml1316[ml1317]

    # L33: ml1311, lower_triangular_udiag, U34: ml1311, upper_triangular, tmp40: ml1318, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1311, ml1318)

    # U34: ml1311, upper_triangular, tmp41: ml1318, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1311, ml1318)

    # tmp17: ml1318, full
    # x = tmp17
    return (ml1318)
end