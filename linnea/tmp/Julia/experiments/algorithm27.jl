using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm27(ml919::Array{Float64,2}, ml920::Array{Float64,2}, ml921::Array{Float64,2}, ml922::Array{Float64,2}, ml923::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml919, full, L: ml920, full, A: ml921, full, B: ml922, full, y: ml923, full
    ml924 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml924, ml922)

    # R: ml919, full, L: ml920, full, A: ml921, full, y: ml923, full, tmp26: ml924, full
    ml925 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml921, ml925, info) = LinearAlgebra.LAPACK.getrf!(ml921)

    # R: ml919, full, L: ml920, full, y: ml923, full, tmp26: ml924, full, P11: ml925, ipiv, L9: ml921, lower_triangular_udiag, U10: ml921, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml921, ml924)

    # R: ml919, full, L: ml920, full, y: ml923, full, P11: ml925, ipiv, L9: ml921, lower_triangular_udiag, tmp27: ml924, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml921, ml924)

    # R: ml919, full, L: ml920, full, y: ml923, full, P11: ml925, ipiv, tmp28: ml924, full
    ml926 = [1:length(ml925);]
    @inbounds for i in 1:length(ml925)
        ml926[i], ml926[ml925[i]] = ml926[ml925[i]], ml926[i];
    end;
    ml927 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml927 = ml924[invperm(ml926),:]

    # R: ml919, full, L: ml920, full, y: ml923, full, tmp25: ml927, full
    ml928 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml927, 0.0, ml928)

    # R: ml919, full, L: ml920, full, y: ml923, full, tmp19: ml928, symmetric_lower_triangular
    ml929 = diag(ml920)
    ml930 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml919, 1, ml930, 1)
    # tmp29 = (L R)
    for i = 1:size(ml919, 2);
        view(ml919, :, i)[:] .*= ml929;
    end;        

    # R: ml930, full, y: ml923, full, tmp19: ml928, symmetric_lower_triangular, tmp29: ml919, full
    for i = 1:2000-1;
        view(ml928, i, i+1:2000)[:] = view(ml928, i+1:2000, i);
    end;
    ml931 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml928, 1, ml931, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml919, ml930, 1.0, ml928)

    # y: ml923, full, tmp19: ml931, full, tmp31: ml928, full
    ml932 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml928, ml932, info) = LinearAlgebra.LAPACK.getrf!(ml928)

    # y: ml923, full, tmp19: ml931, full, P35: ml932, ipiv, L33: ml928, lower_triangular_udiag, U34: ml928, upper_triangular
    ml933 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml931, ml923, 0.0, ml933)

    # P35: ml932, ipiv, L33: ml928, lower_triangular_udiag, U34: ml928, upper_triangular, tmp32: ml933, full
    ml934 = [1:length(ml932);]
    @inbounds for i in 1:length(ml932)
        ml934[i], ml934[ml932[i]] = ml934[ml932[i]], ml934[i];
    end;
    ml935 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml935 = ml933[ml934]

    # L33: ml928, lower_triangular_udiag, U34: ml928, upper_triangular, tmp40: ml935, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml928, ml935)

    # U34: ml928, upper_triangular, tmp41: ml935, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml928, ml935)

    # tmp17: ml935, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml935), (finish-start)*1e-9)
end