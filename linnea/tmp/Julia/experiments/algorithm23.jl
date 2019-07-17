using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm23(ml784::Array{Float64,2}, ml785::Array{Float64,2}, ml786::Array{Float64,2}, ml787::Array{Float64,2}, ml788::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml784, full, L: ml785, full, A: ml786, full, B: ml787, full, y: ml788, full
    ml789 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml786, ml789, info) = LinearAlgebra.LAPACK.getrf!(ml786)

    # R: ml784, full, L: ml785, full, B: ml787, full, y: ml788, full, P11: ml789, ipiv, L9: ml786, lower_triangular_udiag, U10: ml786, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml786, ml787)

    # R: ml784, full, L: ml785, full, y: ml788, full, P11: ml789, ipiv, L9: ml786, lower_triangular_udiag, tmp53: ml787, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml786, ml787)

    # R: ml784, full, L: ml785, full, y: ml788, full, P11: ml789, ipiv, tmp54: ml787, full
    ml790 = [1:length(ml789);]
    @inbounds for i in 1:length(ml789)
        ml790[i], ml790[ml789[i]] = ml790[ml789[i]], ml790[i];
    end;
    ml791 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml791 = ml787[:,invperm(ml790)]

    # R: ml784, full, L: ml785, full, y: ml788, full, tmp55: ml791, full
    ml792 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml792, ml791)

    # R: ml784, full, L: ml785, full, y: ml788, full, tmp25: ml792, full
    ml793 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml792, 0.0, ml793)

    # R: ml784, full, L: ml785, full, y: ml788, full, tmp19: ml793, symmetric_lower_triangular
    ml794 = diag(ml785)
    ml795 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml784, 1, ml795, 1)
    # tmp29 = (L R)
    for i = 1:size(ml784, 2);
        view(ml784, :, i)[:] .*= ml794;
    end;        

    # R: ml795, full, y: ml788, full, tmp19: ml793, symmetric_lower_triangular, tmp29: ml784, full
    ml796 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml793, ml788, 0.0, ml796)

    # R: ml795, full, tmp19: ml793, symmetric_lower_triangular, tmp29: ml784, full, tmp32: ml796, full
    for i = 1:2000-1;
        view(ml793, i, i+1:2000)[:] = view(ml793, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml784, ml795, 1.0, ml793)

    # tmp32: ml796, full, tmp31: ml793, full
    ml797 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml793, ml797, info) = LinearAlgebra.LAPACK.getrf!(ml793)

    # tmp32: ml796, full, P35: ml797, ipiv, L33: ml793, lower_triangular_udiag, U34: ml793, upper_triangular
    ml798 = [1:length(ml797);]
    @inbounds for i in 1:length(ml797)
        ml798[i], ml798[ml797[i]] = ml798[ml797[i]], ml798[i];
    end;
    ml799 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml799 = ml796[ml798]

    # L33: ml793, lower_triangular_udiag, U34: ml793, upper_triangular, tmp40: ml799, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml793, ml799)

    # U34: ml793, upper_triangular, tmp41: ml799, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml793, ml799)

    # tmp17: ml799, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml799), (finish-start)*1e-9)
end