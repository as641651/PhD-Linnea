using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm22(ml752::Array{Float64,2}, ml753::Array{Float64,2}, ml754::Array{Float64,2}, ml755::Array{Float64,2}, ml756::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml752, full, L: ml753, full, A: ml754, full, B: ml755, full, y: ml756, full
    ml757 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml754, ml757, info) = LinearAlgebra.LAPACK.getrf!(ml754)

    # R: ml752, full, L: ml753, full, B: ml755, full, y: ml756, full, P11: ml757, ipiv, L9: ml754, lower_triangular_udiag, U10: ml754, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml754, ml755)

    # R: ml752, full, L: ml753, full, y: ml756, full, P11: ml757, ipiv, L9: ml754, lower_triangular_udiag, tmp53: ml755, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml754, ml755)

    # R: ml752, full, L: ml753, full, y: ml756, full, P11: ml757, ipiv, tmp54: ml755, full
    ml758 = [1:length(ml757);]
    @inbounds for i in 1:length(ml757)
        ml758[i], ml758[ml757[i]] = ml758[ml757[i]], ml758[i];
    end;
    ml759 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml759 = ml755[:,invperm(ml758)]

    # R: ml752, full, L: ml753, full, y: ml756, full, tmp55: ml759, full
    ml760 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml760, ml759)

    # R: ml752, full, L: ml753, full, y: ml756, full, tmp25: ml760, full
    ml761 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml760, 0.0, ml761)

    # R: ml752, full, L: ml753, full, y: ml756, full, tmp19: ml761, symmetric_lower_triangular
    ml762 = diag(ml753)
    ml763 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml752, 1, ml763, 1)
    # tmp29 = (L R)
    for i = 1:size(ml752, 2);
        view(ml752, :, i)[:] .*= ml762;
    end;        

    # R: ml763, full, y: ml756, full, tmp19: ml761, symmetric_lower_triangular, tmp29: ml752, full
    ml764 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml761, ml756, 0.0, ml764)

    # R: ml763, full, tmp19: ml761, symmetric_lower_triangular, tmp29: ml752, full, tmp32: ml764, full
    for i = 1:2000-1;
        view(ml761, i, i+1:2000)[:] = view(ml761, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml752, ml763, 1.0, ml761)

    # tmp32: ml764, full, tmp31: ml761, full
    ml765 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml761, ml765, info) = LinearAlgebra.LAPACK.getrf!(ml761)

    # tmp32: ml764, full, P35: ml765, ipiv, L33: ml761, lower_triangular_udiag, U34: ml761, upper_triangular
    ml766 = [1:length(ml765);]
    @inbounds for i in 1:length(ml765)
        ml766[i], ml766[ml765[i]] = ml766[ml765[i]], ml766[i];
    end;
    ml767 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml767 = ml764[ml766]

    # L33: ml761, lower_triangular_udiag, U34: ml761, upper_triangular, tmp40: ml767, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml761, ml767)

    # U34: ml761, upper_triangular, tmp41: ml767, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml761, ml767)

    # tmp17: ml767, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml767), (finish-start)*1e-9)
end