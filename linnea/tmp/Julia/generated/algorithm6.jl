using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm6(ml196::Array{Float64,2}, ml197::Array{Float64,2}, ml198::Array{Float64,2}, ml199::Array{Float64,2}, ml200::Array{Float64,1})
    # cost 5.07e+10
    # R: ml196, full, L: ml197, full, A: ml198, full, B: ml199, full, y: ml200, full
    ml201 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml198, ml201, info) = LinearAlgebra.LAPACK.getrf!(ml198)

    # R: ml196, full, L: ml197, full, B: ml199, full, y: ml200, full, P11: ml201, ipiv, L9: ml198, lower_triangular_udiag, U10: ml198, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml198, ml199)

    # R: ml196, full, L: ml197, full, y: ml200, full, P11: ml201, ipiv, L9: ml198, lower_triangular_udiag, tmp53: ml199, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml198, ml199)

    # R: ml196, full, L: ml197, full, y: ml200, full, P11: ml201, ipiv, tmp54: ml199, full
    ml202 = [1:length(ml201);]
    @inbounds for i in 1:length(ml201)
        ml202[i], ml202[ml201[i]] = ml202[ml201[i]], ml202[i];
    end;
    ml203 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml203 = ml199[:,invperm(ml202)]

    # R: ml196, full, L: ml197, full, y: ml200, full, tmp55: ml203, full
    ml204 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml204, ml203)

    # R: ml196, full, L: ml197, full, y: ml200, full, tmp25: ml204, full
    ml205 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml204, 0.0, ml205)

    # R: ml196, full, L: ml197, full, y: ml200, full, tmp19: ml205, symmetric_lower_triangular
    ml206 = diag(ml197)
    ml207 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml196, 1, ml207, 1)
    # tmp29 = (L R)
    for i = 1:size(ml196, 2);
        view(ml196, :, i)[:] .*= ml206;
    end;        

    # R: ml207, full, y: ml200, full, tmp19: ml205, symmetric_lower_triangular, tmp29: ml196, full
    for i = 1:2000-1;
        view(ml205, i, i+1:2000)[:] = view(ml205, i+1:2000, i);
    end;
    ml208 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml205, 1, ml208, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml196, ml207, 1.0, ml205)

    # y: ml200, full, tmp19: ml208, full, tmp31: ml205, full
    ml209 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml205, ml209, info) = LinearAlgebra.LAPACK.getrf!(ml205)

    # y: ml200, full, tmp19: ml208, full, P35: ml209, ipiv, L33: ml205, lower_triangular_udiag, U34: ml205, upper_triangular
    ml210 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml208, ml200, 0.0, ml210)

    # P35: ml209, ipiv, L33: ml205, lower_triangular_udiag, U34: ml205, upper_triangular, tmp32: ml210, full
    ml211 = [1:length(ml209);]
    @inbounds for i in 1:length(ml209)
        ml211[i], ml211[ml209[i]] = ml211[ml209[i]], ml211[i];
    end;
    ml212 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml212 = ml210[ml211]

    # L33: ml205, lower_triangular_udiag, U34: ml205, upper_triangular, tmp40: ml212, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml205, ml212)

    # U34: ml205, upper_triangular, tmp41: ml212, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml205, ml212)

    # tmp17: ml212, full
    # x = tmp17
    return (ml212)
end