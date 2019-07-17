using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm94(ml3140::Array{Float64,2}, ml3141::Array{Float64,2}, ml3142::Array{Float64,2}, ml3143::Array{Float64,2}, ml3144::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3140, full, L: ml3141, full, A: ml3142, full, B: ml3143, full, y: ml3144, full
    ml3145 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3145, ml3143)

    # R: ml3140, full, L: ml3141, full, A: ml3142, full, y: ml3144, full, tmp26: ml3145, full
    ml3146 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3142, ml3146, info) = LinearAlgebra.LAPACK.getrf!(ml3142)

    # R: ml3140, full, L: ml3141, full, y: ml3144, full, tmp26: ml3145, full, P11: ml3146, ipiv, L9: ml3142, lower_triangular_udiag, U10: ml3142, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3142, ml3145)

    # R: ml3140, full, L: ml3141, full, y: ml3144, full, P11: ml3146, ipiv, L9: ml3142, lower_triangular_udiag, tmp27: ml3145, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3142, ml3145)

    # R: ml3140, full, L: ml3141, full, y: ml3144, full, P11: ml3146, ipiv, tmp28: ml3145, full
    ml3147 = [1:length(ml3146);]
    @inbounds for i in 1:length(ml3146)
        ml3147[i], ml3147[ml3146[i]] = ml3147[ml3146[i]], ml3147[i];
    end;
    ml3148 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3148 = ml3145[invperm(ml3147),:]

    # R: ml3140, full, L: ml3141, full, y: ml3144, full, tmp25: ml3148, full
    ml3149 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3148, 0.0, ml3149)

    # R: ml3140, full, L: ml3141, full, y: ml3144, full, tmp19: ml3149, symmetric_lower_triangular
    ml3150 = diag(ml3141)
    ml3151 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3140, 1, ml3151, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3140, 2);
        view(ml3140, :, i)[:] .*= ml3150;
    end;        

    # R: ml3151, full, y: ml3144, full, tmp19: ml3149, symmetric_lower_triangular, tmp29: ml3140, full
    for i = 1:2000-1;
        view(ml3149, i, i+1:2000)[:] = view(ml3149, i+1:2000, i);
    end;
    ml3152 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml3149, 1, ml3152, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml3140, ml3151, 1.0, ml3149)

    # y: ml3144, full, tmp19: ml3152, full, tmp31: ml3149, full
    ml3153 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3149, ml3153, info) = LinearAlgebra.LAPACK.getrf!(ml3149)

    # y: ml3144, full, tmp19: ml3152, full, P35: ml3153, ipiv, L33: ml3149, lower_triangular_udiag, U34: ml3149, upper_triangular
    ml3154 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3152, ml3144, 0.0, ml3154)

    # P35: ml3153, ipiv, L33: ml3149, lower_triangular_udiag, U34: ml3149, upper_triangular, tmp32: ml3154, full
    ml3155 = [1:length(ml3153);]
    @inbounds for i in 1:length(ml3153)
        ml3155[i], ml3155[ml3153[i]] = ml3155[ml3153[i]], ml3155[i];
    end;
    ml3156 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3156 = ml3154[ml3155]

    # L33: ml3149, lower_triangular_udiag, U34: ml3149, upper_triangular, tmp40: ml3156, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3149, ml3156)

    # U34: ml3149, upper_triangular, tmp41: ml3156, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3149, ml3156)

    # tmp17: ml3156, full
    # x = tmp17
    return (ml3156)
end