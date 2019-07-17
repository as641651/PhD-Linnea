using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm28(ml953::Array{Float64,2}, ml954::Array{Float64,2}, ml955::Array{Float64,2}, ml956::Array{Float64,2}, ml957::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml953, full, L: ml954, full, A: ml955, full, B: ml956, full, y: ml957, full
    ml958 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml958, ml956)

    # R: ml953, full, L: ml954, full, A: ml955, full, y: ml957, full, tmp26: ml958, full
    ml959 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml955, ml959, info) = LinearAlgebra.LAPACK.getrf!(ml955)

    # R: ml953, full, L: ml954, full, y: ml957, full, tmp26: ml958, full, P11: ml959, ipiv, L9: ml955, lower_triangular_udiag, U10: ml955, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml955, ml958)

    # R: ml953, full, L: ml954, full, y: ml957, full, P11: ml959, ipiv, L9: ml955, lower_triangular_udiag, tmp27: ml958, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml955, ml958)

    # R: ml953, full, L: ml954, full, y: ml957, full, P11: ml959, ipiv, tmp28: ml958, full
    ml960 = [1:length(ml959);]
    @inbounds for i in 1:length(ml959)
        ml960[i], ml960[ml959[i]] = ml960[ml959[i]], ml960[i];
    end;
    ml961 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml961 = ml958[invperm(ml960),:]

    # R: ml953, full, L: ml954, full, y: ml957, full, tmp25: ml961, full
    ml962 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml961, 0.0, ml962)

    # R: ml953, full, L: ml954, full, y: ml957, full, tmp19: ml962, symmetric_lower_triangular
    ml963 = diag(ml954)
    ml964 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml953, 1, ml964, 1)
    # tmp29 = (L R)
    for i = 1:size(ml953, 2);
        view(ml953, :, i)[:] .*= ml963;
    end;        

    # R: ml964, full, y: ml957, full, tmp19: ml962, symmetric_lower_triangular, tmp29: ml953, full
    for i = 1:2000-1;
        view(ml962, i, i+1:2000)[:] = view(ml962, i+1:2000, i);
    end;
    ml965 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml962, 1, ml965, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml953, ml964, 1.0, ml962)

    # y: ml957, full, tmp19: ml965, full, tmp31: ml962, full
    ml966 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml962, ml966, info) = LinearAlgebra.LAPACK.getrf!(ml962)

    # y: ml957, full, tmp19: ml965, full, P35: ml966, ipiv, L33: ml962, lower_triangular_udiag, U34: ml962, upper_triangular
    ml967 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml965, ml957, 0.0, ml967)

    # P35: ml966, ipiv, L33: ml962, lower_triangular_udiag, U34: ml962, upper_triangular, tmp32: ml967, full
    ml968 = [1:length(ml966);]
    @inbounds for i in 1:length(ml966)
        ml968[i], ml968[ml966[i]] = ml968[ml966[i]], ml968[i];
    end;
    ml969 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml969 = ml967[ml968]

    # L33: ml962, lower_triangular_udiag, U34: ml962, upper_triangular, tmp40: ml969, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml962, ml969)

    # U34: ml962, upper_triangular, tmp41: ml969, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml962, ml969)

    # tmp17: ml969, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml969), (finish-start)*1e-9)
end