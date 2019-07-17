using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm24(ml817::Array{Float64,2}, ml818::Array{Float64,2}, ml819::Array{Float64,2}, ml820::Array{Float64,2}, ml821::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml817, full, L: ml818, full, A: ml819, full, B: ml820, full, y: ml821, full
    ml822 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml822, ml820)

    # R: ml817, full, L: ml818, full, A: ml819, full, y: ml821, full, tmp26: ml822, full
    ml823 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml819, ml823, info) = LinearAlgebra.LAPACK.getrf!(ml819)

    # R: ml817, full, L: ml818, full, y: ml821, full, tmp26: ml822, full, P11: ml823, ipiv, L9: ml819, lower_triangular_udiag, U10: ml819, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml819, ml822)

    # R: ml817, full, L: ml818, full, y: ml821, full, P11: ml823, ipiv, L9: ml819, lower_triangular_udiag, tmp27: ml822, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml819, ml822)

    # R: ml817, full, L: ml818, full, y: ml821, full, P11: ml823, ipiv, tmp28: ml822, full
    ml824 = [1:length(ml823);]
    @inbounds for i in 1:length(ml823)
        ml824[i], ml824[ml823[i]] = ml824[ml823[i]], ml824[i];
    end;
    ml825 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml825 = ml822[invperm(ml824),:]

    # R: ml817, full, L: ml818, full, y: ml821, full, tmp25: ml825, full
    ml826 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml825, 0.0, ml826)

    # R: ml817, full, L: ml818, full, y: ml821, full, tmp19: ml826, symmetric_lower_triangular
    ml827 = diag(ml818)
    ml828 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml817, 1, ml828, 1)
    # tmp29 = (L R)
    for i = 1:size(ml817, 2);
        view(ml817, :, i)[:] .*= ml827;
    end;        

    # R: ml828, full, y: ml821, full, tmp19: ml826, symmetric_lower_triangular, tmp29: ml817, full
    for i = 1:2000-1;
        view(ml826, i, i+1:2000)[:] = view(ml826, i+1:2000, i);
    end;
    ml829 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml826, 1, ml829, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml817, ml828, 1.0, ml826)

    # y: ml821, full, tmp19: ml829, full, tmp31: ml826, full
    ml830 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml826, ml830, info) = LinearAlgebra.LAPACK.getrf!(ml826)

    # y: ml821, full, tmp19: ml829, full, P35: ml830, ipiv, L33: ml826, lower_triangular_udiag, U34: ml826, upper_triangular
    ml831 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml829, ml821, 0.0, ml831)

    # P35: ml830, ipiv, L33: ml826, lower_triangular_udiag, U34: ml826, upper_triangular, tmp32: ml831, full
    ml832 = [1:length(ml830);]
    @inbounds for i in 1:length(ml830)
        ml832[i], ml832[ml830[i]] = ml832[ml830[i]], ml832[i];
    end;
    ml833 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml833 = ml831[ml832]

    # L33: ml826, lower_triangular_udiag, U34: ml826, upper_triangular, tmp40: ml833, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml826, ml833)

    # U34: ml826, upper_triangular, tmp41: ml833, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml826, ml833)

    # tmp17: ml833, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml833), (finish-start)*1e-9)
end