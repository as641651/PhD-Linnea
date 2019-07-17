using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm7(ml230::Array{Float64,2}, ml231::Array{Float64,2}, ml232::Array{Float64,2}, ml233::Array{Float64,2}, ml234::Array{Float64,1})
    # cost 5.07e+10
    # R: ml230, full, L: ml231, full, A: ml232, full, B: ml233, full, y: ml234, full
    ml235 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml232, ml235, info) = LinearAlgebra.LAPACK.getrf!(ml232)

    # R: ml230, full, L: ml231, full, B: ml233, full, y: ml234, full, P11: ml235, ipiv, L9: ml232, lower_triangular_udiag, U10: ml232, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml232, ml233)

    # R: ml230, full, L: ml231, full, y: ml234, full, P11: ml235, ipiv, L9: ml232, lower_triangular_udiag, tmp53: ml233, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml232, ml233)

    # R: ml230, full, L: ml231, full, y: ml234, full, P11: ml235, ipiv, tmp54: ml233, full
    ml236 = [1:length(ml235);]
    @inbounds for i in 1:length(ml235)
        ml236[i], ml236[ml235[i]] = ml236[ml235[i]], ml236[i];
    end;
    ml237 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml237 = ml233[:,invperm(ml236)]

    # R: ml230, full, L: ml231, full, y: ml234, full, tmp55: ml237, full
    ml238 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml238, ml237)

    # R: ml230, full, L: ml231, full, y: ml234, full, tmp25: ml238, full
    ml239 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml238, 0.0, ml239)

    # R: ml230, full, L: ml231, full, y: ml234, full, tmp19: ml239, symmetric_lower_triangular
    ml240 = diag(ml231)
    ml241 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml230, 1, ml241, 1)
    # tmp29 = (L R)
    for i = 1:size(ml230, 2);
        view(ml230, :, i)[:] .*= ml240;
    end;        

    # R: ml241, full, y: ml234, full, tmp19: ml239, symmetric_lower_triangular, tmp29: ml230, full
    for i = 1:2000-1;
        view(ml239, i, i+1:2000)[:] = view(ml239, i+1:2000, i);
    end;
    ml242 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml239, 1, ml242, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml230, ml241, 1.0, ml239)

    # y: ml234, full, tmp19: ml242, full, tmp31: ml239, full
    ml243 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml239, ml243, info) = LinearAlgebra.LAPACK.getrf!(ml239)

    # y: ml234, full, tmp19: ml242, full, P35: ml243, ipiv, L33: ml239, lower_triangular_udiag, U34: ml239, upper_triangular
    ml244 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml242, ml234, 0.0, ml244)

    # P35: ml243, ipiv, L33: ml239, lower_triangular_udiag, U34: ml239, upper_triangular, tmp32: ml244, full
    ml245 = [1:length(ml243);]
    @inbounds for i in 1:length(ml243)
        ml245[i], ml245[ml243[i]] = ml245[ml243[i]], ml245[i];
    end;
    ml246 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml246 = ml244[ml245]

    # L33: ml239, lower_triangular_udiag, U34: ml239, upper_triangular, tmp40: ml246, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml239, ml246)

    # U34: ml239, upper_triangular, tmp41: ml246, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml239, ml246)

    # tmp17: ml246, full
    # x = tmp17
    return (ml246)
end