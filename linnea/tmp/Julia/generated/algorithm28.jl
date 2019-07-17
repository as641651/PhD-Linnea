using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm28(ml936::Array{Float64,2}, ml937::Array{Float64,2}, ml938::Array{Float64,2}, ml939::Array{Float64,2}, ml940::Array{Float64,1})
    # cost 5.07e+10
    # R: ml936, full, L: ml937, full, A: ml938, full, B: ml939, full, y: ml940, full
    ml941 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml941, ml939)

    # R: ml936, full, L: ml937, full, A: ml938, full, y: ml940, full, tmp26: ml941, full
    ml942 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml938, ml942, info) = LinearAlgebra.LAPACK.getrf!(ml938)

    # R: ml936, full, L: ml937, full, y: ml940, full, tmp26: ml941, full, P11: ml942, ipiv, L9: ml938, lower_triangular_udiag, U10: ml938, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml938, ml941)

    # R: ml936, full, L: ml937, full, y: ml940, full, P11: ml942, ipiv, L9: ml938, lower_triangular_udiag, tmp27: ml941, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml938, ml941)

    # R: ml936, full, L: ml937, full, y: ml940, full, P11: ml942, ipiv, tmp28: ml941, full
    ml943 = [1:length(ml942);]
    @inbounds for i in 1:length(ml942)
        ml943[i], ml943[ml942[i]] = ml943[ml942[i]], ml943[i];
    end;
    ml944 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml944 = ml941[invperm(ml943),:]

    # R: ml936, full, L: ml937, full, y: ml940, full, tmp25: ml944, full
    ml945 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml944, 0.0, ml945)

    # R: ml936, full, L: ml937, full, y: ml940, full, tmp19: ml945, symmetric_lower_triangular
    ml946 = diag(ml937)
    ml947 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml936, 1, ml947, 1)
    # tmp29 = (L R)
    for i = 1:size(ml936, 2);
        view(ml936, :, i)[:] .*= ml946;
    end;        

    # R: ml947, full, y: ml940, full, tmp19: ml945, symmetric_lower_triangular, tmp29: ml936, full
    for i = 1:2000-1;
        view(ml945, i, i+1:2000)[:] = view(ml945, i+1:2000, i);
    end;
    ml948 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml945, 1, ml948, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml936, ml947, 1.0, ml945)

    # y: ml940, full, tmp19: ml948, full, tmp31: ml945, full
    ml949 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml945, ml949, info) = LinearAlgebra.LAPACK.getrf!(ml945)

    # y: ml940, full, tmp19: ml948, full, P35: ml949, ipiv, L33: ml945, lower_triangular_udiag, U34: ml945, upper_triangular
    ml950 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml948, ml940, 0.0, ml950)

    # P35: ml949, ipiv, L33: ml945, lower_triangular_udiag, U34: ml945, upper_triangular, tmp32: ml950, full
    ml951 = [1:length(ml949);]
    @inbounds for i in 1:length(ml949)
        ml951[i], ml951[ml949[i]] = ml951[ml949[i]], ml951[i];
    end;
    ml952 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml952 = ml950[ml951]

    # L33: ml945, lower_triangular_udiag, U34: ml945, upper_triangular, tmp40: ml952, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml945, ml952)

    # U34: ml945, upper_triangular, tmp41: ml952, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml945, ml952)

    # tmp17: ml952, full
    # x = tmp17
    return (ml952)
end