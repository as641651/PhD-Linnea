using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm76(ml2544::Array{Float64,2}, ml2545::Array{Float64,2}, ml2546::Array{Float64,2}, ml2547::Array{Float64,2}, ml2548::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2544, full, L: ml2545, full, A: ml2546, full, B: ml2547, full, y: ml2548, full
    ml2549 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2549, ml2547)

    # R: ml2544, full, L: ml2545, full, A: ml2546, full, y: ml2548, full, tmp26: ml2549, full
    ml2550 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2546, ml2550, info) = LinearAlgebra.LAPACK.getrf!(ml2546)

    # R: ml2544, full, L: ml2545, full, y: ml2548, full, tmp26: ml2549, full, P11: ml2550, ipiv, L9: ml2546, lower_triangular_udiag, U10: ml2546, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2546, ml2549)

    # R: ml2544, full, L: ml2545, full, y: ml2548, full, P11: ml2550, ipiv, L9: ml2546, lower_triangular_udiag, tmp27: ml2549, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2546, ml2549)

    # R: ml2544, full, L: ml2545, full, y: ml2548, full, P11: ml2550, ipiv, tmp28: ml2549, full
    ml2551 = [1:length(ml2550);]
    @inbounds for i in 1:length(ml2550)
        ml2551[i], ml2551[ml2550[i]] = ml2551[ml2550[i]], ml2551[i];
    end;
    ml2552 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2552 = ml2549[invperm(ml2551),:]

    # R: ml2544, full, L: ml2545, full, y: ml2548, full, tmp25: ml2552, full
    ml2553 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2552, 0.0, ml2553)

    # R: ml2544, full, L: ml2545, full, y: ml2548, full, tmp19: ml2553, symmetric_lower_triangular
    ml2554 = diag(ml2545)
    ml2555 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2544, 1, ml2555, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2544, 2);
        view(ml2544, :, i)[:] .*= ml2554;
    end;        

    # R: ml2555, full, y: ml2548, full, tmp19: ml2553, symmetric_lower_triangular, tmp29: ml2544, full
    for i = 1:2000-1;
        view(ml2553, i, i+1:2000)[:] = view(ml2553, i+1:2000, i);
    end;
    ml2556 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2553, 1, ml2556, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2544, ml2555, 1.0, ml2553)

    # y: ml2548, full, tmp19: ml2556, full, tmp31: ml2553, full
    ml2557 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2553, ml2557, info) = LinearAlgebra.LAPACK.getrf!(ml2553)

    # y: ml2548, full, tmp19: ml2556, full, P35: ml2557, ipiv, L33: ml2553, lower_triangular_udiag, U34: ml2553, upper_triangular
    ml2558 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2556, ml2548, 0.0, ml2558)

    # P35: ml2557, ipiv, L33: ml2553, lower_triangular_udiag, U34: ml2553, upper_triangular, tmp32: ml2558, full
    ml2559 = [1:length(ml2557);]
    @inbounds for i in 1:length(ml2557)
        ml2559[i], ml2559[ml2557[i]] = ml2559[ml2557[i]], ml2559[i];
    end;
    ml2560 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2560 = ml2558[ml2559]

    # L33: ml2553, lower_triangular_udiag, U34: ml2553, upper_triangular, tmp40: ml2560, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2553, ml2560)

    # U34: ml2553, upper_triangular, tmp41: ml2560, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2553, ml2560)

    # tmp17: ml2560, full
    # x = tmp17
    return (ml2560)
end