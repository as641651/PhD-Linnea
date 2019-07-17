using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm4(ml128::Array{Float64,2}, ml129::Array{Float64,2}, ml130::Array{Float64,2}, ml131::Array{Float64,2}, ml132::Array{Float64,1})
    # cost 5.07e+10
    # R: ml128, full, L: ml129, full, A: ml130, full, B: ml131, full, y: ml132, full
    ml133 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml130, ml133, info) = LinearAlgebra.LAPACK.getrf!(ml130)

    # R: ml128, full, L: ml129, full, B: ml131, full, y: ml132, full, P11: ml133, ipiv, L9: ml130, lower_triangular_udiag, U10: ml130, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml130, ml131)

    # R: ml128, full, L: ml129, full, y: ml132, full, P11: ml133, ipiv, L9: ml130, lower_triangular_udiag, tmp53: ml131, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml130, ml131)

    # R: ml128, full, L: ml129, full, y: ml132, full, P11: ml133, ipiv, tmp54: ml131, full
    ml134 = [1:length(ml133);]
    @inbounds for i in 1:length(ml133)
        ml134[i], ml134[ml133[i]] = ml134[ml133[i]], ml134[i];
    end;
    ml135 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml135 = ml131[:,invperm(ml134)]

    # R: ml128, full, L: ml129, full, y: ml132, full, tmp55: ml135, full
    ml136 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml136, ml135)

    # R: ml128, full, L: ml129, full, y: ml132, full, tmp25: ml136, full
    ml137 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml136, 0.0, ml137)

    # R: ml128, full, L: ml129, full, y: ml132, full, tmp19: ml137, symmetric_lower_triangular
    ml138 = diag(ml129)
    ml139 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml128, 1, ml139, 1)
    # tmp29 = (L R)
    for i = 1:size(ml128, 2);
        view(ml128, :, i)[:] .*= ml138;
    end;        

    # R: ml139, full, y: ml132, full, tmp19: ml137, symmetric_lower_triangular, tmp29: ml128, full
    for i = 1:2000-1;
        view(ml137, i, i+1:2000)[:] = view(ml137, i+1:2000, i);
    end;
    ml140 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml137, 1, ml140, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml128, ml139, 1.0, ml137)

    # y: ml132, full, tmp19: ml140, full, tmp31: ml137, full
    ml141 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml137, ml141, info) = LinearAlgebra.LAPACK.getrf!(ml137)

    # y: ml132, full, tmp19: ml140, full, P35: ml141, ipiv, L33: ml137, lower_triangular_udiag, U34: ml137, upper_triangular
    ml142 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml140, ml132, 0.0, ml142)

    # P35: ml141, ipiv, L33: ml137, lower_triangular_udiag, U34: ml137, upper_triangular, tmp32: ml142, full
    ml143 = [1:length(ml141);]
    @inbounds for i in 1:length(ml141)
        ml143[i], ml143[ml141[i]] = ml143[ml141[i]], ml143[i];
    end;
    ml144 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml144 = ml142[ml143]

    # L33: ml137, lower_triangular_udiag, U34: ml137, upper_triangular, tmp40: ml144, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml137, ml144)

    # U34: ml137, upper_triangular, tmp41: ml144, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml137, ml144)

    # tmp17: ml144, full
    # x = tmp17
    return (ml144)
end