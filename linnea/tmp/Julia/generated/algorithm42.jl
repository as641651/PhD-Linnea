using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm42(ml1404::Array{Float64,2}, ml1405::Array{Float64,2}, ml1406::Array{Float64,2}, ml1407::Array{Float64,2}, ml1408::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1404, full, L: ml1405, full, A: ml1406, full, B: ml1407, full, y: ml1408, full
    ml1409 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1409, ml1407)

    # R: ml1404, full, L: ml1405, full, A: ml1406, full, y: ml1408, full, tmp26: ml1409, full
    ml1410 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1406, ml1410, info) = LinearAlgebra.LAPACK.getrf!(ml1406)

    # R: ml1404, full, L: ml1405, full, y: ml1408, full, tmp26: ml1409, full, P11: ml1410, ipiv, L9: ml1406, lower_triangular_udiag, U10: ml1406, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1406, ml1409)

    # R: ml1404, full, L: ml1405, full, y: ml1408, full, P11: ml1410, ipiv, L9: ml1406, lower_triangular_udiag, tmp27: ml1409, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1406, ml1409)

    # R: ml1404, full, L: ml1405, full, y: ml1408, full, P11: ml1410, ipiv, tmp28: ml1409, full
    ml1411 = [1:length(ml1410);]
    @inbounds for i in 1:length(ml1410)
        ml1411[i], ml1411[ml1410[i]] = ml1411[ml1410[i]], ml1411[i];
    end;
    ml1412 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1412 = ml1409[invperm(ml1411),:]

    # R: ml1404, full, L: ml1405, full, y: ml1408, full, tmp25: ml1412, full
    ml1413 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1412, 0.0, ml1413)

    # R: ml1404, full, L: ml1405, full, y: ml1408, full, tmp19: ml1413, symmetric_lower_triangular
    ml1414 = diag(ml1405)
    ml1415 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1404, 1, ml1415, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1404, 2);
        view(ml1404, :, i)[:] .*= ml1414;
    end;        

    # R: ml1415, full, y: ml1408, full, tmp19: ml1413, symmetric_lower_triangular, tmp29: ml1404, full
    for i = 1:2000-1;
        view(ml1413, i, i+1:2000)[:] = view(ml1413, i+1:2000, i);
    end;
    ml1416 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1413, 1, ml1416, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1404, ml1415, 1.0, ml1413)

    # y: ml1408, full, tmp19: ml1416, full, tmp31: ml1413, full
    ml1417 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1416, ml1408, 0.0, ml1417)

    # tmp31: ml1413, full, tmp32: ml1417, full
    ml1418 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1413, ml1418, info) = LinearAlgebra.LAPACK.getrf!(ml1413)

    # tmp32: ml1417, full, P35: ml1418, ipiv, L33: ml1413, lower_triangular_udiag, U34: ml1413, upper_triangular
    ml1419 = [1:length(ml1418);]
    @inbounds for i in 1:length(ml1418)
        ml1419[i], ml1419[ml1418[i]] = ml1419[ml1418[i]], ml1419[i];
    end;
    ml1420 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1420 = ml1417[ml1419]

    # L33: ml1413, lower_triangular_udiag, U34: ml1413, upper_triangular, tmp40: ml1420, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1413, ml1420)

    # U34: ml1413, upper_triangular, tmp41: ml1420, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1413, ml1420)

    # tmp17: ml1420, full
    # x = tmp17
    return (ml1420)
end