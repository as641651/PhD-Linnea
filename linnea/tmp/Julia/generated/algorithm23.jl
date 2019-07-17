using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm23(ml768::Array{Float64,2}, ml769::Array{Float64,2}, ml770::Array{Float64,2}, ml771::Array{Float64,2}, ml772::Array{Float64,1})
    # cost 5.07e+10
    # R: ml768, full, L: ml769, full, A: ml770, full, B: ml771, full, y: ml772, full
    ml773 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml770, ml773, info) = LinearAlgebra.LAPACK.getrf!(ml770)

    # R: ml768, full, L: ml769, full, B: ml771, full, y: ml772, full, P11: ml773, ipiv, L9: ml770, lower_triangular_udiag, U10: ml770, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml770, ml771)

    # R: ml768, full, L: ml769, full, y: ml772, full, P11: ml773, ipiv, L9: ml770, lower_triangular_udiag, tmp53: ml771, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml770, ml771)

    # R: ml768, full, L: ml769, full, y: ml772, full, P11: ml773, ipiv, tmp54: ml771, full
    ml774 = [1:length(ml773);]
    @inbounds for i in 1:length(ml773)
        ml774[i], ml774[ml773[i]] = ml774[ml773[i]], ml774[i];
    end;
    ml775 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml775 = ml771[:,invperm(ml774)]

    # R: ml768, full, L: ml769, full, y: ml772, full, tmp55: ml775, full
    ml776 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml776, ml775)

    # R: ml768, full, L: ml769, full, y: ml772, full, tmp25: ml776, full
    ml777 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml776, 0.0, ml777)

    # R: ml768, full, L: ml769, full, y: ml772, full, tmp19: ml777, symmetric_lower_triangular
    ml778 = diag(ml769)
    ml779 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml768, 1, ml779, 1)
    # tmp29 = (L R)
    for i = 1:size(ml768, 2);
        view(ml768, :, i)[:] .*= ml778;
    end;        

    # R: ml779, full, y: ml772, full, tmp19: ml777, symmetric_lower_triangular, tmp29: ml768, full
    ml780 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml777, ml772, 0.0, ml780)

    # R: ml779, full, tmp19: ml777, symmetric_lower_triangular, tmp29: ml768, full, tmp32: ml780, full
    for i = 1:2000-1;
        view(ml777, i, i+1:2000)[:] = view(ml777, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml768, ml779, 1.0, ml777)

    # tmp32: ml780, full, tmp31: ml777, full
    ml781 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml777, ml781, info) = LinearAlgebra.LAPACK.getrf!(ml777)

    # tmp32: ml780, full, P35: ml781, ipiv, L33: ml777, lower_triangular_udiag, U34: ml777, upper_triangular
    ml782 = [1:length(ml781);]
    @inbounds for i in 1:length(ml781)
        ml782[i], ml782[ml781[i]] = ml782[ml781[i]], ml782[i];
    end;
    ml783 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml783 = ml780[ml782]

    # L33: ml777, lower_triangular_udiag, U34: ml777, upper_triangular, tmp40: ml783, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml777, ml783)

    # U34: ml777, upper_triangular, tmp41: ml783, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml777, ml783)

    # tmp17: ml783, full
    # x = tmp17
    return (ml783)
end