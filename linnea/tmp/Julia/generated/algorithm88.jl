using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm88(ml2944::Array{Float64,2}, ml2945::Array{Float64,2}, ml2946::Array{Float64,2}, ml2947::Array{Float64,2}, ml2948::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2944, full, L: ml2945, full, A: ml2946, full, B: ml2947, full, y: ml2948, full
    ml2949 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2949, ml2947)

    # R: ml2944, full, L: ml2945, full, A: ml2946, full, y: ml2948, full, tmp26: ml2949, full
    ml2950 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2946, ml2950, info) = LinearAlgebra.LAPACK.getrf!(ml2946)

    # R: ml2944, full, L: ml2945, full, y: ml2948, full, tmp26: ml2949, full, P11: ml2950, ipiv, L9: ml2946, lower_triangular_udiag, U10: ml2946, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2946, ml2949)

    # R: ml2944, full, L: ml2945, full, y: ml2948, full, P11: ml2950, ipiv, L9: ml2946, lower_triangular_udiag, tmp27: ml2949, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2946, ml2949)

    # R: ml2944, full, L: ml2945, full, y: ml2948, full, P11: ml2950, ipiv, tmp28: ml2949, full
    ml2951 = [1:length(ml2950);]
    @inbounds for i in 1:length(ml2950)
        ml2951[i], ml2951[ml2950[i]] = ml2951[ml2950[i]], ml2951[i];
    end;
    ml2952 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2952 = ml2949[invperm(ml2951),:]

    # R: ml2944, full, L: ml2945, full, y: ml2948, full, tmp25: ml2952, full
    ml2953 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2952, 0.0, ml2953)

    # R: ml2944, full, L: ml2945, full, y: ml2948, full, tmp19: ml2953, symmetric_lower_triangular
    ml2954 = diag(ml2945)
    ml2955 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2944, 1, ml2955, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2944, 2);
        view(ml2944, :, i)[:] .*= ml2954;
    end;        

    # R: ml2955, full, y: ml2948, full, tmp19: ml2953, symmetric_lower_triangular, tmp29: ml2944, full
    for i = 1:2000-1;
        view(ml2953, i, i+1:2000)[:] = view(ml2953, i+1:2000, i);
    end;
    ml2956 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml2953, 1, ml2956, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml2944, ml2955, 1.0, ml2953)

    # y: ml2948, full, tmp19: ml2956, full, tmp31: ml2953, full
    ml2957 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2953, ml2957, info) = LinearAlgebra.LAPACK.getrf!(ml2953)

    # y: ml2948, full, tmp19: ml2956, full, P35: ml2957, ipiv, L33: ml2953, lower_triangular_udiag, U34: ml2953, upper_triangular
    ml2958 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2956, ml2948, 0.0, ml2958)

    # P35: ml2957, ipiv, L33: ml2953, lower_triangular_udiag, U34: ml2953, upper_triangular, tmp32: ml2958, full
    ml2959 = [1:length(ml2957);]
    @inbounds for i in 1:length(ml2957)
        ml2959[i], ml2959[ml2957[i]] = ml2959[ml2957[i]], ml2959[i];
    end;
    ml2960 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2960 = ml2958[ml2959]

    # L33: ml2953, lower_triangular_udiag, U34: ml2953, upper_triangular, tmp40: ml2960, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2953, ml2960)

    # U34: ml2953, upper_triangular, tmp41: ml2960, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2953, ml2960)

    # tmp17: ml2960, full
    # x = tmp17
    return (ml2960)
end