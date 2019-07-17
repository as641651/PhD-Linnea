using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm96(ml3208::Array{Float64,2}, ml3209::Array{Float64,2}, ml3210::Array{Float64,2}, ml3211::Array{Float64,2}, ml3212::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3208, full, L: ml3209, full, A: ml3210, full, B: ml3211, full, y: ml3212, full
    ml3213 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3213, ml3211)

    # R: ml3208, full, L: ml3209, full, A: ml3210, full, y: ml3212, full, tmp26: ml3213, full
    ml3214 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3210, ml3214, info) = LinearAlgebra.LAPACK.getrf!(ml3210)

    # R: ml3208, full, L: ml3209, full, y: ml3212, full, tmp26: ml3213, full, P11: ml3214, ipiv, L9: ml3210, lower_triangular_udiag, U10: ml3210, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3210, ml3213)

    # R: ml3208, full, L: ml3209, full, y: ml3212, full, P11: ml3214, ipiv, L9: ml3210, lower_triangular_udiag, tmp27: ml3213, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3210, ml3213)

    # R: ml3208, full, L: ml3209, full, y: ml3212, full, P11: ml3214, ipiv, tmp28: ml3213, full
    ml3215 = [1:length(ml3214);]
    @inbounds for i in 1:length(ml3214)
        ml3215[i], ml3215[ml3214[i]] = ml3215[ml3214[i]], ml3215[i];
    end;
    ml3216 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3216 = ml3213[invperm(ml3215),:]

    # R: ml3208, full, L: ml3209, full, y: ml3212, full, tmp25: ml3216, full
    ml3217 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3216, 0.0, ml3217)

    # R: ml3208, full, L: ml3209, full, y: ml3212, full, tmp19: ml3217, symmetric_lower_triangular
    ml3218 = diag(ml3209)
    ml3219 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3208, 1, ml3219, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3208, 2);
        view(ml3208, :, i)[:] .*= ml3218;
    end;        

    # R: ml3219, full, y: ml3212, full, tmp19: ml3217, symmetric_lower_triangular, tmp29: ml3208, full
    for i = 1:2000-1;
        view(ml3217, i, i+1:2000)[:] = view(ml3217, i+1:2000, i);
    end;
    ml3220 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3217, 1, ml3220, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3208, ml3219, 1.0, ml3217)

    # y: ml3212, full, tmp19: ml3220, full, tmp31: ml3217, full
    ml3221 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3217, ml3221, info) = LinearAlgebra.LAPACK.getrf!(ml3217)

    # y: ml3212, full, tmp19: ml3220, full, P35: ml3221, ipiv, L33: ml3217, lower_triangular_udiag, U34: ml3217, upper_triangular
    ml3222 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3220, ml3212, 0.0, ml3222)

    # P35: ml3221, ipiv, L33: ml3217, lower_triangular_udiag, U34: ml3217, upper_triangular, tmp32: ml3222, full
    ml3223 = [1:length(ml3221);]
    @inbounds for i in 1:length(ml3221)
        ml3223[i], ml3223[ml3221[i]] = ml3223[ml3221[i]], ml3223[i];
    end;
    ml3224 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3224 = ml3222[ml3223]

    # L33: ml3217, lower_triangular_udiag, U34: ml3217, upper_triangular, tmp40: ml3224, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3217, ml3224)

    # U34: ml3217, upper_triangular, tmp41: ml3224, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3217, ml3224)

    # tmp17: ml3224, full
    # x = tmp17
    return (ml3224)
end