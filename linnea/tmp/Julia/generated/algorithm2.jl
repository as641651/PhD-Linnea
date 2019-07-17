using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm2(ml64::Array{Float64,2}, ml65::Array{Float64,2}, ml66::Array{Float64,2}, ml67::Array{Float64,2}, ml68::Array{Float64,1})
    # cost 5.07e+10
    # R: ml64, full, L: ml65, full, A: ml66, full, B: ml67, full, y: ml68, full
    ml69 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml66, ml69, info) = LinearAlgebra.LAPACK.getrf!(ml66)

    # R: ml64, full, L: ml65, full, B: ml67, full, y: ml68, full, P11: ml69, ipiv, L9: ml66, lower_triangular_udiag, U10: ml66, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml66, ml67)

    # R: ml64, full, L: ml65, full, y: ml68, full, P11: ml69, ipiv, L9: ml66, lower_triangular_udiag, tmp53: ml67, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml66, ml67)

    # R: ml64, full, L: ml65, full, y: ml68, full, P11: ml69, ipiv, tmp54: ml67, full
    ml70 = [1:length(ml69);]
    @inbounds for i in 1:length(ml69)
        ml70[i], ml70[ml69[i]] = ml70[ml69[i]], ml70[i];
    end;
    ml71 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml71 = ml67[:,invperm(ml70)]

    # R: ml64, full, L: ml65, full, y: ml68, full, tmp55: ml71, full
    ml72 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml72, ml71)

    # R: ml64, full, L: ml65, full, y: ml68, full, tmp25: ml72, full
    ml73 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml72, 0.0, ml73)

    # R: ml64, full, L: ml65, full, y: ml68, full, tmp19: ml73, symmetric_lower_triangular
    ml74 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml73, ml68, 0.0, ml74)

    # R: ml64, full, L: ml65, full, tmp19: ml73, symmetric_lower_triangular, tmp32: ml74, full
    ml75 = diag(ml65)
    ml76 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml64, 1, ml76, 1)
    # tmp29 = (L R)
    for i = 1:size(ml64, 2);
        view(ml64, :, i)[:] .*= ml75;
    end;        

    # R: ml76, full, tmp19: ml73, symmetric_lower_triangular, tmp32: ml74, full, tmp29: ml64, full
    for i = 1:2000-1;
        view(ml73, i, i+1:2000)[:] = view(ml73, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml76, ml64, 1.0, ml73)

    # tmp32: ml74, full, tmp31: ml73, full
    ml77 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml73, ml77, info) = LinearAlgebra.LAPACK.getrf!(ml73)

    # tmp32: ml74, full, P35: ml77, ipiv, L33: ml73, lower_triangular_udiag, U34: ml73, upper_triangular
    ml78 = [1:length(ml77);]
    @inbounds for i in 1:length(ml77)
        ml78[i], ml78[ml77[i]] = ml78[ml77[i]], ml78[i];
    end;
    ml79 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml79 = ml74[ml78]

    # L33: ml73, lower_triangular_udiag, U34: ml73, upper_triangular, tmp40: ml79, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml73, ml79)

    # U34: ml73, upper_triangular, tmp41: ml79, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml73, ml79)

    # tmp17: ml79, full
    # x = tmp17
    return (ml79)
end