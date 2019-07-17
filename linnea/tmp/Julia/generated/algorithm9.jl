using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm9(ml298::Array{Float64,2}, ml299::Array{Float64,2}, ml300::Array{Float64,2}, ml301::Array{Float64,2}, ml302::Array{Float64,1})
    # cost 5.07e+10
    # R: ml298, full, L: ml299, full, A: ml300, full, B: ml301, full, y: ml302, full
    ml303 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml303, ml301)

    # R: ml298, full, L: ml299, full, A: ml300, full, y: ml302, full, tmp26: ml303, full
    ml304 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml300, ml304, info) = LinearAlgebra.LAPACK.getrf!(ml300)

    # R: ml298, full, L: ml299, full, y: ml302, full, tmp26: ml303, full, P11: ml304, ipiv, L9: ml300, lower_triangular_udiag, U10: ml300, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml300, ml303)

    # R: ml298, full, L: ml299, full, y: ml302, full, P11: ml304, ipiv, L9: ml300, lower_triangular_udiag, tmp27: ml303, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml300, ml303)

    # R: ml298, full, L: ml299, full, y: ml302, full, P11: ml304, ipiv, tmp28: ml303, full
    ml305 = [1:length(ml304);]
    @inbounds for i in 1:length(ml304)
        ml305[i], ml305[ml304[i]] = ml305[ml304[i]], ml305[i];
    end;
    ml306 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml306 = ml303[invperm(ml305),:]

    # R: ml298, full, L: ml299, full, y: ml302, full, tmp25: ml306, full
    ml307 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml306, 0.0, ml307)

    # R: ml298, full, L: ml299, full, y: ml302, full, tmp19: ml307, symmetric_lower_triangular
    ml308 = diag(ml299)
    ml309 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml298, 1, ml309, 1)
    # tmp29 = (L R)
    for i = 1:size(ml298, 2);
        view(ml298, :, i)[:] .*= ml308;
    end;        

    # R: ml309, full, y: ml302, full, tmp19: ml307, symmetric_lower_triangular, tmp29: ml298, full
    for i = 1:2000-1;
        view(ml307, i, i+1:2000)[:] = view(ml307, i+1:2000, i);
    end;
    ml310 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml307, 1, ml310, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml298, ml309, 1.0, ml307)

    # y: ml302, full, tmp19: ml310, full, tmp31: ml307, full
    ml311 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml307, ml311, info) = LinearAlgebra.LAPACK.getrf!(ml307)

    # y: ml302, full, tmp19: ml310, full, P35: ml311, ipiv, L33: ml307, lower_triangular_udiag, U34: ml307, upper_triangular
    ml312 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml310, ml302, 0.0, ml312)

    # P35: ml311, ipiv, L33: ml307, lower_triangular_udiag, U34: ml307, upper_triangular, tmp32: ml312, full
    ml313 = [1:length(ml311);]
    @inbounds for i in 1:length(ml311)
        ml313[i], ml313[ml311[i]] = ml313[ml311[i]], ml313[i];
    end;
    ml314 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml314 = ml312[ml313]

    # L33: ml307, lower_triangular_udiag, U34: ml307, upper_triangular, tmp40: ml314, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml307, ml314)

    # U34: ml307, upper_triangular, tmp41: ml314, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml307, ml314)

    # tmp17: ml314, full
    # x = tmp17
    return (ml314)
end