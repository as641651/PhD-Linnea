using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm12(ml400::Array{Float64,2}, ml401::Array{Float64,2}, ml402::Array{Float64,2}, ml403::Array{Float64,2}, ml404::Array{Float64,1})
    # cost 5.07e+10
    # R: ml400, full, L: ml401, full, A: ml402, full, B: ml403, full, y: ml404, full
    ml405 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml402, ml405, info) = LinearAlgebra.LAPACK.getrf!(ml402)

    # R: ml400, full, L: ml401, full, B: ml403, full, y: ml404, full, P11: ml405, ipiv, L9: ml402, lower_triangular_udiag, U10: ml402, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml402, ml403)

    # R: ml400, full, L: ml401, full, y: ml404, full, P11: ml405, ipiv, L9: ml402, lower_triangular_udiag, tmp53: ml403, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml402, ml403)

    # R: ml400, full, L: ml401, full, y: ml404, full, P11: ml405, ipiv, tmp54: ml403, full
    ml406 = [1:length(ml405);]
    @inbounds for i in 1:length(ml405)
        ml406[i], ml406[ml405[i]] = ml406[ml405[i]], ml406[i];
    end;
    ml407 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml407 = ml403[:,invperm(ml406)]

    # R: ml400, full, L: ml401, full, y: ml404, full, tmp55: ml407, full
    ml408 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml408, ml407)

    # R: ml400, full, L: ml401, full, y: ml404, full, tmp25: ml408, full
    ml409 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml408, 0.0, ml409)

    # R: ml400, full, L: ml401, full, y: ml404, full, tmp19: ml409, symmetric_lower_triangular
    ml410 = diag(ml401)
    ml411 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml400, 1, ml411, 1)
    # tmp29 = (L R)
    for i = 1:size(ml400, 2);
        view(ml400, :, i)[:] .*= ml410;
    end;        

    # R: ml411, full, y: ml404, full, tmp19: ml409, symmetric_lower_triangular, tmp29: ml400, full
    for i = 1:2000-1;
        view(ml409, i, i+1:2000)[:] = view(ml409, i+1:2000, i);
    end;
    ml412 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml409, 1, ml412, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml400, ml411, 1.0, ml409)

    # y: ml404, full, tmp19: ml412, full, tmp31: ml409, full
    ml413 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml412, ml404, 0.0, ml413)

    # tmp31: ml409, full, tmp32: ml413, full
    ml414 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml409, ml414, info) = LinearAlgebra.LAPACK.getrf!(ml409)

    # tmp32: ml413, full, P35: ml414, ipiv, L33: ml409, lower_triangular_udiag, U34: ml409, upper_triangular
    ml415 = [1:length(ml414);]
    @inbounds for i in 1:length(ml414)
        ml415[i], ml415[ml414[i]] = ml415[ml414[i]], ml415[i];
    end;
    ml416 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml416 = ml413[ml415]

    # L33: ml409, lower_triangular_udiag, U34: ml409, upper_triangular, tmp40: ml416, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml409, ml416)

    # U34: ml409, upper_triangular, tmp41: ml416, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml409, ml416)

    # tmp17: ml416, full
    # x = tmp17
    return (ml416)
end