using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm38(ml1285::Array{Float64,2}, ml1286::Array{Float64,2}, ml1287::Array{Float64,2}, ml1288::Array{Float64,2}, ml1289::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml1285, full, L: ml1286, full, A: ml1287, full, B: ml1288, full, y: ml1289, full
    ml1290 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml1290, ml1288)

    # R: ml1285, full, L: ml1286, full, A: ml1287, full, y: ml1289, full, tmp26: ml1290, full
    ml1291 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1287, ml1291, info) = LinearAlgebra.LAPACK.getrf!(ml1287)

    # R: ml1285, full, L: ml1286, full, y: ml1289, full, tmp26: ml1290, full, P11: ml1291, ipiv, L9: ml1287, lower_triangular_udiag, U10: ml1287, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml1287, ml1290)

    # R: ml1285, full, L: ml1286, full, y: ml1289, full, P11: ml1291, ipiv, L9: ml1287, lower_triangular_udiag, tmp27: ml1290, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml1287, ml1290)

    # R: ml1285, full, L: ml1286, full, y: ml1289, full, P11: ml1291, ipiv, tmp28: ml1290, full
    ml1292 = [1:length(ml1291);]
    @inbounds for i in 1:length(ml1291)
        ml1292[i], ml1292[ml1291[i]] = ml1292[ml1291[i]], ml1292[i];
    end;
    ml1293 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml1293 = ml1290[invperm(ml1292),:]

    # R: ml1285, full, L: ml1286, full, y: ml1289, full, tmp25: ml1293, full
    ml1294 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1293, 0.0, ml1294)

    # R: ml1285, full, L: ml1286, full, y: ml1289, full, tmp19: ml1294, symmetric_lower_triangular
    ml1295 = diag(ml1286)
    ml1296 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1285, 1, ml1296, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1285, 2);
        view(ml1285, :, i)[:] .*= ml1295;
    end;        

    # R: ml1296, full, y: ml1289, full, tmp19: ml1294, symmetric_lower_triangular, tmp29: ml1285, full
    for i = 1:2000-1;
        view(ml1294, i, i+1:2000)[:] = view(ml1294, i+1:2000, i);
    end;
    ml1297 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1294, 1, ml1297, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1285, ml1296, 1.0, ml1294)

    # y: ml1289, full, tmp19: ml1297, full, tmp31: ml1294, full
    ml1298 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1294, ml1298, info) = LinearAlgebra.LAPACK.getrf!(ml1294)

    # y: ml1289, full, tmp19: ml1297, full, P35: ml1298, ipiv, L33: ml1294, lower_triangular_udiag, U34: ml1294, upper_triangular
    ml1299 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1297, ml1289, 0.0, ml1299)

    # P35: ml1298, ipiv, L33: ml1294, lower_triangular_udiag, U34: ml1294, upper_triangular, tmp32: ml1299, full
    ml1300 = [1:length(ml1298);]
    @inbounds for i in 1:length(ml1298)
        ml1300[i], ml1300[ml1298[i]] = ml1300[ml1298[i]], ml1300[i];
    end;
    ml1301 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1301 = ml1299[ml1300]

    # L33: ml1294, lower_triangular_udiag, U34: ml1294, upper_triangular, tmp40: ml1301, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1294, ml1301)

    # U34: ml1294, upper_triangular, tmp41: ml1301, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1294, ml1301)

    # tmp17: ml1301, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml1301), (finish-start)*1e-9)
end