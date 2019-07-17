using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm90(ml3012::Array{Float64,2}, ml3013::Array{Float64,2}, ml3014::Array{Float64,2}, ml3015::Array{Float64,2}, ml3016::Array{Float64,1})
    # cost 5.07e+10
    # R: ml3012, full, L: ml3013, full, A: ml3014, full, B: ml3015, full, y: ml3016, full
    ml3017 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml3017, ml3015)

    # R: ml3012, full, L: ml3013, full, A: ml3014, full, y: ml3016, full, tmp26: ml3017, full
    ml3018 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml3014, ml3018, info) = LinearAlgebra.LAPACK.getrf!(ml3014)

    # R: ml3012, full, L: ml3013, full, y: ml3016, full, tmp26: ml3017, full, P11: ml3018, ipiv, L9: ml3014, lower_triangular_udiag, U10: ml3014, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml3014, ml3017)

    # R: ml3012, full, L: ml3013, full, y: ml3016, full, P11: ml3018, ipiv, L9: ml3014, lower_triangular_udiag, tmp27: ml3017, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml3014, ml3017)

    # R: ml3012, full, L: ml3013, full, y: ml3016, full, P11: ml3018, ipiv, tmp28: ml3017, full
    ml3019 = [1:length(ml3018);]
    @inbounds for i in 1:length(ml3018)
        ml3019[i], ml3019[ml3018[i]] = ml3019[ml3018[i]], ml3019[i];
    end;
    ml3020 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml3020 = ml3017[invperm(ml3019),:]

    # R: ml3012, full, L: ml3013, full, y: ml3016, full, tmp25: ml3020, full
    ml3021 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml3020, 0.0, ml3021)

    # R: ml3012, full, L: ml3013, full, y: ml3016, full, tmp19: ml3021, symmetric_lower_triangular
    ml3022 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml3021, ml3016, 0.0, ml3022)

    # R: ml3012, full, L: ml3013, full, tmp19: ml3021, symmetric_lower_triangular, tmp32: ml3022, full
    ml3023 = diag(ml3013)
    ml3024 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml3012, 1, ml3024, 1)
    # tmp29 = (L R)
    for i = 1:size(ml3012, 2);
        view(ml3012, :, i)[:] .*= ml3023;
    end;        

    # R: ml3024, full, tmp19: ml3021, symmetric_lower_triangular, tmp32: ml3022, full, tmp29: ml3012, full
    for i = 1:2000-1;
        view(ml3021, i, i+1:2000)[:] = view(ml3021, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml3024, ml3012, 1.0, ml3021)

    # tmp32: ml3022, full, tmp31: ml3021, full
    ml3025 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml3021, ml3025, info) = LinearAlgebra.LAPACK.getrf!(ml3021)

    # tmp32: ml3022, full, P35: ml3025, ipiv, L33: ml3021, lower_triangular_udiag, U34: ml3021, upper_triangular
    ml3026 = [1:length(ml3025);]
    @inbounds for i in 1:length(ml3025)
        ml3026[i], ml3026[ml3025[i]] = ml3026[ml3025[i]], ml3026[i];
    end;
    ml3027 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml3027 = ml3022[ml3026]

    # L33: ml3021, lower_triangular_udiag, U34: ml3021, upper_triangular, tmp40: ml3027, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml3021, ml3027)

    # U34: ml3021, upper_triangular, tmp41: ml3027, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml3021, ml3027)

    # tmp17: ml3027, full
    # x = tmp17
    return (ml3027)
end