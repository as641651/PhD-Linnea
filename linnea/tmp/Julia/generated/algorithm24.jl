using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm24(ml800::Array{Float64,2}, ml801::Array{Float64,2}, ml802::Array{Float64,2}, ml803::Array{Float64,2}, ml804::Array{Float64,1})
    # cost 5.07e+10
    # R: ml800, full, L: ml801, full, A: ml802, full, B: ml803, full, y: ml804, full
    ml805 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml805, ml803)

    # R: ml800, full, L: ml801, full, A: ml802, full, y: ml804, full, tmp26: ml805, full
    ml806 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml802, ml806, info) = LinearAlgebra.LAPACK.getrf!(ml802)

    # R: ml800, full, L: ml801, full, y: ml804, full, tmp26: ml805, full, P11: ml806, ipiv, L9: ml802, lower_triangular_udiag, U10: ml802, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml802, ml805)

    # R: ml800, full, L: ml801, full, y: ml804, full, P11: ml806, ipiv, L9: ml802, lower_triangular_udiag, tmp27: ml805, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml802, ml805)

    # R: ml800, full, L: ml801, full, y: ml804, full, P11: ml806, ipiv, tmp28: ml805, full
    ml807 = [1:length(ml806);]
    @inbounds for i in 1:length(ml806)
        ml807[i], ml807[ml806[i]] = ml807[ml806[i]], ml807[i];
    end;
    ml808 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml808 = ml805[invperm(ml807),:]

    # R: ml800, full, L: ml801, full, y: ml804, full, tmp25: ml808, full
    ml809 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml808, 0.0, ml809)

    # R: ml800, full, L: ml801, full, y: ml804, full, tmp19: ml809, symmetric_lower_triangular
    ml810 = diag(ml801)
    ml811 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml800, 1, ml811, 1)
    # tmp29 = (L R)
    for i = 1:size(ml800, 2);
        view(ml800, :, i)[:] .*= ml810;
    end;        

    # R: ml811, full, y: ml804, full, tmp19: ml809, symmetric_lower_triangular, tmp29: ml800, full
    for i = 1:2000-1;
        view(ml809, i, i+1:2000)[:] = view(ml809, i+1:2000, i);
    end;
    ml812 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml809, 1, ml812, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml800, ml811, 1.0, ml809)

    # y: ml804, full, tmp19: ml812, full, tmp31: ml809, full
    ml813 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml809, ml813, info) = LinearAlgebra.LAPACK.getrf!(ml809)

    # y: ml804, full, tmp19: ml812, full, P35: ml813, ipiv, L33: ml809, lower_triangular_udiag, U34: ml809, upper_triangular
    ml814 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml812, ml804, 0.0, ml814)

    # P35: ml813, ipiv, L33: ml809, lower_triangular_udiag, U34: ml809, upper_triangular, tmp32: ml814, full
    ml815 = [1:length(ml813);]
    @inbounds for i in 1:length(ml813)
        ml815[i], ml815[ml813[i]] = ml815[ml813[i]], ml815[i];
    end;
    ml816 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml816 = ml814[ml815]

    # L33: ml809, lower_triangular_udiag, U34: ml809, upper_triangular, tmp40: ml816, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml809, ml816)

    # U34: ml809, upper_triangular, tmp41: ml816, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml809, ml816)

    # tmp17: ml816, full
    # x = tmp17
    return (ml816)
end