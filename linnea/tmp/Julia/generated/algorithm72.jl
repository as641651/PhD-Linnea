using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm72(ml2408::Array{Float64,2}, ml2409::Array{Float64,2}, ml2410::Array{Float64,2}, ml2411::Array{Float64,2}, ml2412::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2408, full, L: ml2409, full, A: ml2410, full, B: ml2411, full, y: ml2412, full
    ml2413 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2413, ml2411)

    # R: ml2408, full, L: ml2409, full, A: ml2410, full, y: ml2412, full, tmp26: ml2413, full
    ml2414 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2410, ml2414, info) = LinearAlgebra.LAPACK.getrf!(ml2410)

    # R: ml2408, full, L: ml2409, full, y: ml2412, full, tmp26: ml2413, full, P11: ml2414, ipiv, L9: ml2410, lower_triangular_udiag, U10: ml2410, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2410, ml2413)

    # R: ml2408, full, L: ml2409, full, y: ml2412, full, P11: ml2414, ipiv, L9: ml2410, lower_triangular_udiag, tmp27: ml2413, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2410, ml2413)

    # R: ml2408, full, L: ml2409, full, y: ml2412, full, P11: ml2414, ipiv, tmp28: ml2413, full
    ml2415 = [1:length(ml2414);]
    @inbounds for i in 1:length(ml2414)
        ml2415[i], ml2415[ml2414[i]] = ml2415[ml2414[i]], ml2415[i];
    end;
    ml2416 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2416 = ml2413[invperm(ml2415),:]

    # R: ml2408, full, L: ml2409, full, y: ml2412, full, tmp25: ml2416, full
    ml2417 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2416, 0.0, ml2417)

    # R: ml2408, full, L: ml2409, full, y: ml2412, full, tmp19: ml2417, symmetric_lower_triangular
    ml2418 = diag(ml2409)
    ml2419 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2408, 1, ml2419, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2408, 2);
        view(ml2408, :, i)[:] .*= ml2418;
    end;        

    # R: ml2419, full, y: ml2412, full, tmp19: ml2417, symmetric_lower_triangular, tmp29: ml2408, full
    for i = 1:2000-1;
        view(ml2417, i, i+1:2000)[:] = view(ml2417, i+1:2000, i);
    end;
    ml2420 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2417, 1, ml2420, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2408, ml2419, 1.0, ml2417)

    # y: ml2412, full, tmp19: ml2420, full, tmp31: ml2417, full
    ml2421 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2420, ml2412, 0.0, ml2421)

    # tmp31: ml2417, full, tmp32: ml2421, full
    ml2422 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2417, ml2422, info) = LinearAlgebra.LAPACK.getrf!(ml2417)

    # tmp32: ml2421, full, P35: ml2422, ipiv, L33: ml2417, lower_triangular_udiag, U34: ml2417, upper_triangular
    ml2423 = [1:length(ml2422);]
    @inbounds for i in 1:length(ml2422)
        ml2423[i], ml2423[ml2422[i]] = ml2423[ml2422[i]], ml2423[i];
    end;
    ml2424 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2424 = ml2421[ml2423]

    # L33: ml2417, lower_triangular_udiag, U34: ml2417, upper_triangular, tmp40: ml2424, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2417, ml2424)

    # U34: ml2417, upper_triangular, tmp41: ml2424, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2417, ml2424)

    # tmp17: ml2424, full
    # x = tmp17
    return (ml2424)
end