using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm16(ml536::Array{Float64,2}, ml537::Array{Float64,2}, ml538::Array{Float64,2}, ml539::Array{Float64,2}, ml540::Array{Float64,1})
    # cost 5.07e+10
    # R: ml536, full, L: ml537, full, A: ml538, full, B: ml539, full, y: ml540, full
    ml541 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml541, ml539)

    # R: ml536, full, L: ml537, full, A: ml538, full, y: ml540, full, tmp26: ml541, full
    ml542 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml538, ml542, info) = LinearAlgebra.LAPACK.getrf!(ml538)

    # R: ml536, full, L: ml537, full, y: ml540, full, tmp26: ml541, full, P11: ml542, ipiv, L9: ml538, lower_triangular_udiag, U10: ml538, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml538, ml541)

    # R: ml536, full, L: ml537, full, y: ml540, full, P11: ml542, ipiv, L9: ml538, lower_triangular_udiag, tmp27: ml541, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml538, ml541)

    # R: ml536, full, L: ml537, full, y: ml540, full, P11: ml542, ipiv, tmp28: ml541, full
    ml543 = [1:length(ml542);]
    @inbounds for i in 1:length(ml542)
        ml543[i], ml543[ml542[i]] = ml543[ml542[i]], ml543[i];
    end;
    ml544 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml544 = ml541[invperm(ml543),:]

    # R: ml536, full, L: ml537, full, y: ml540, full, tmp25: ml544, full
    ml545 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml544, 0.0, ml545)

    # R: ml536, full, L: ml537, full, y: ml540, full, tmp19: ml545, symmetric_lower_triangular
    ml546 = diag(ml537)
    ml547 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml536, 1, ml547, 1)
    # tmp29 = (L R)
    for i = 1:size(ml536, 2);
        view(ml536, :, i)[:] .*= ml546;
    end;        

    # R: ml547, full, y: ml540, full, tmp19: ml545, symmetric_lower_triangular, tmp29: ml536, full
    for i = 1:2000-1;
        view(ml545, i, i+1:2000)[:] = view(ml545, i+1:2000, i);
    end;
    ml548 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml545, 1, ml548, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml536, ml547, 1.0, ml545)

    # y: ml540, full, tmp19: ml548, full, tmp31: ml545, full
    ml549 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml545, ml549, info) = LinearAlgebra.LAPACK.getrf!(ml545)

    # y: ml540, full, tmp19: ml548, full, P35: ml549, ipiv, L33: ml545, lower_triangular_udiag, U34: ml545, upper_triangular
    ml550 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml548, ml540, 0.0, ml550)

    # P35: ml549, ipiv, L33: ml545, lower_triangular_udiag, U34: ml545, upper_triangular, tmp32: ml550, full
    ml551 = [1:length(ml549);]
    @inbounds for i in 1:length(ml549)
        ml551[i], ml551[ml549[i]] = ml551[ml549[i]], ml551[i];
    end;
    ml552 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml552 = ml550[ml551]

    # L33: ml545, lower_triangular_udiag, U34: ml545, upper_triangular, tmp40: ml552, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml545, ml552)

    # U34: ml545, upper_triangular, tmp41: ml552, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml545, ml552)

    # tmp17: ml552, full
    # x = tmp17
    return (ml552)
end