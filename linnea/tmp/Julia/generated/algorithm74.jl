using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm74(ml2476::Array{Float64,2}, ml2477::Array{Float64,2}, ml2478::Array{Float64,2}, ml2479::Array{Float64,2}, ml2480::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2476, full, L: ml2477, full, A: ml2478, full, B: ml2479, full, y: ml2480, full
    ml2481 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2481, ml2479)

    # R: ml2476, full, L: ml2477, full, A: ml2478, full, y: ml2480, full, tmp26: ml2481, full
    ml2482 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2478, ml2482, info) = LinearAlgebra.LAPACK.getrf!(ml2478)

    # R: ml2476, full, L: ml2477, full, y: ml2480, full, tmp26: ml2481, full, P11: ml2482, ipiv, L9: ml2478, lower_triangular_udiag, U10: ml2478, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2478, ml2481)

    # R: ml2476, full, L: ml2477, full, y: ml2480, full, P11: ml2482, ipiv, L9: ml2478, lower_triangular_udiag, tmp27: ml2481, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2478, ml2481)

    # R: ml2476, full, L: ml2477, full, y: ml2480, full, P11: ml2482, ipiv, tmp28: ml2481, full
    ml2483 = [1:length(ml2482);]
    @inbounds for i in 1:length(ml2482)
        ml2483[i], ml2483[ml2482[i]] = ml2483[ml2482[i]], ml2483[i];
    end;
    ml2484 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2484 = ml2481[invperm(ml2483),:]

    # R: ml2476, full, L: ml2477, full, y: ml2480, full, tmp25: ml2484, full
    ml2485 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2484, 0.0, ml2485)

    # R: ml2476, full, L: ml2477, full, y: ml2480, full, tmp19: ml2485, symmetric_lower_triangular
    ml2486 = diag(ml2477)
    ml2487 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2476, 1, ml2487, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2476, 2);
        view(ml2476, :, i)[:] .*= ml2486;
    end;        

    # R: ml2487, full, y: ml2480, full, tmp19: ml2485, symmetric_lower_triangular, tmp29: ml2476, full
    for i = 1:2000-1;
        view(ml2485, i, i+1:2000)[:] = view(ml2485, i+1:2000, i);
    end;
    ml2488 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2485, 1, ml2488, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2476, ml2487, 1.0, ml2485)

    # y: ml2480, full, tmp19: ml2488, full, tmp31: ml2485, full
    ml2489 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2485, ml2489, info) = LinearAlgebra.LAPACK.getrf!(ml2485)

    # y: ml2480, full, tmp19: ml2488, full, P35: ml2489, ipiv, L33: ml2485, lower_triangular_udiag, U34: ml2485, upper_triangular
    ml2490 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2488, ml2480, 0.0, ml2490)

    # P35: ml2489, ipiv, L33: ml2485, lower_triangular_udiag, U34: ml2485, upper_triangular, tmp32: ml2490, full
    ml2491 = [1:length(ml2489);]
    @inbounds for i in 1:length(ml2489)
        ml2491[i], ml2491[ml2489[i]] = ml2491[ml2489[i]], ml2491[i];
    end;
    ml2492 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2492 = ml2490[ml2491]

    # L33: ml2485, lower_triangular_udiag, U34: ml2485, upper_triangular, tmp40: ml2492, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2485, ml2492)

    # U34: ml2485, upper_triangular, tmp41: ml2492, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2485, ml2492)

    # tmp17: ml2492, full
    # x = tmp17
    return (ml2492)
end