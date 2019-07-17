using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm26(ml885::Array{Float64,2}, ml886::Array{Float64,2}, ml887::Array{Float64,2}, ml888::Array{Float64,2}, ml889::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml885, full, L: ml886, full, A: ml887, full, B: ml888, full, y: ml889, full
    ml890 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml890, ml888)

    # R: ml885, full, L: ml886, full, A: ml887, full, y: ml889, full, tmp26: ml890, full
    ml891 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml887, ml891, info) = LinearAlgebra.LAPACK.getrf!(ml887)

    # R: ml885, full, L: ml886, full, y: ml889, full, tmp26: ml890, full, P11: ml891, ipiv, L9: ml887, lower_triangular_udiag, U10: ml887, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml887, ml890)

    # R: ml885, full, L: ml886, full, y: ml889, full, P11: ml891, ipiv, L9: ml887, lower_triangular_udiag, tmp27: ml890, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml887, ml890)

    # R: ml885, full, L: ml886, full, y: ml889, full, P11: ml891, ipiv, tmp28: ml890, full
    ml892 = [1:length(ml891);]
    @inbounds for i in 1:length(ml891)
        ml892[i], ml892[ml891[i]] = ml892[ml891[i]], ml892[i];
    end;
    ml893 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml893 = ml890[invperm(ml892),:]

    # R: ml885, full, L: ml886, full, y: ml889, full, tmp25: ml893, full
    ml894 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml893, 0.0, ml894)

    # R: ml885, full, L: ml886, full, y: ml889, full, tmp19: ml894, symmetric_lower_triangular
    ml895 = diag(ml886)
    ml896 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml885, 1, ml896, 1)
    # tmp29 = (L R)
    for i = 1:size(ml885, 2);
        view(ml885, :, i)[:] .*= ml895;
    end;        

    # R: ml896, full, y: ml889, full, tmp19: ml894, symmetric_lower_triangular, tmp29: ml885, full
    for i = 1:2000-1;
        view(ml894, i, i+1:2000)[:] = view(ml894, i+1:2000, i);
    end;
    ml897 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml894, 1, ml897, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml885, ml896, 1.0, ml894)

    # y: ml889, full, tmp19: ml897, full, tmp31: ml894, full
    ml898 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml894, ml898, info) = LinearAlgebra.LAPACK.getrf!(ml894)

    # y: ml889, full, tmp19: ml897, full, P35: ml898, ipiv, L33: ml894, lower_triangular_udiag, U34: ml894, upper_triangular
    ml899 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml897, ml889, 0.0, ml899)

    # P35: ml898, ipiv, L33: ml894, lower_triangular_udiag, U34: ml894, upper_triangular, tmp32: ml899, full
    ml900 = [1:length(ml898);]
    @inbounds for i in 1:length(ml898)
        ml900[i], ml900[ml898[i]] = ml900[ml898[i]], ml900[i];
    end;
    ml901 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml901 = ml899[ml900]

    # L33: ml894, lower_triangular_udiag, U34: ml894, upper_triangular, tmp40: ml901, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml894, ml901)

    # U34: ml894, upper_triangular, tmp41: ml901, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml894, ml901)

    # tmp17: ml901, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml901), (finish-start)*1e-9)
end