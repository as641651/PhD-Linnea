using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm40(ml1336::Array{Float64,2}, ml1337::Array{Float64,2}, ml1338::Array{Float64,2}, ml1339::Array{Float64,2}, ml1340::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1336, full, L: ml1337, full, A: ml1338, full, B: ml1339, full, y: ml1340, full
    ml1341 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1341, ml1339)

    # R: ml1336, full, L: ml1337, full, A: ml1338, full, y: ml1340, full, tmp26: ml1341, full
    ml1342 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1338, ml1342, info) = LinearAlgebra.LAPACK.getrf!(ml1338)

    # R: ml1336, full, L: ml1337, full, y: ml1340, full, tmp26: ml1341, full, P11: ml1342, ipiv, L9: ml1338, lower_triangular_udiag, U10: ml1338, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1338, ml1341)

    # R: ml1336, full, L: ml1337, full, y: ml1340, full, P11: ml1342, ipiv, L9: ml1338, lower_triangular_udiag, tmp27: ml1341, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1338, ml1341)

    # R: ml1336, full, L: ml1337, full, y: ml1340, full, P11: ml1342, ipiv, tmp28: ml1341, full
    ml1343 = [1:length(ml1342);]
    @inbounds for i in 1:length(ml1342)
        ml1343[i], ml1343[ml1342[i]] = ml1343[ml1342[i]], ml1343[i];
    end;
    ml1344 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1344 = ml1341[invperm(ml1343),:]

    # R: ml1336, full, L: ml1337, full, y: ml1340, full, tmp25: ml1344, full
    ml1345 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1344, 0.0, ml1345)

    # R: ml1336, full, L: ml1337, full, y: ml1340, full, tmp19: ml1345, symmetric_lower_triangular
    ml1346 = diag(ml1337)
    ml1347 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1336, 1, ml1347, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1336, 2);
        view(ml1336, :, i)[:] .*= ml1346;
    end;        

    # R: ml1347, full, y: ml1340, full, tmp19: ml1345, symmetric_lower_triangular, tmp29: ml1336, full
    for i = 1:2000-1;
        view(ml1345, i, i+1:2000)[:] = view(ml1345, i+1:2000, i);
    end;
    ml1348 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1345, 1, ml1348, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1336, ml1347, 1.0, ml1345)

    # y: ml1340, full, tmp19: ml1348, full, tmp31: ml1345, full
    ml1349 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1348, ml1340, 0.0, ml1349)

    # tmp31: ml1345, full, tmp32: ml1349, full
    ml1350 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1345, ml1350, info) = LinearAlgebra.LAPACK.getrf!(ml1345)

    # tmp32: ml1349, full, P35: ml1350, ipiv, L33: ml1345, lower_triangular_udiag, U34: ml1345, upper_triangular
    ml1351 = [1:length(ml1350);]
    @inbounds for i in 1:length(ml1350)
        ml1351[i], ml1351[ml1350[i]] = ml1351[ml1350[i]], ml1351[i];
    end;
    ml1352 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1352 = ml1349[ml1351]

    # L33: ml1345, lower_triangular_udiag, U34: ml1345, upper_triangular, tmp40: ml1352, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1345, ml1352)

    # U34: ml1345, upper_triangular, tmp41: ml1352, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1345, ml1352)

    # tmp17: ml1352, full
    # x = tmp17
    return (ml1352)
end