using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm4(ml145::Array{Float64,2}, ml146::Array{Float64,2}, ml147::Array{Float64,2}, ml148::Array{Float64,2}, ml149::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    # cost 5.07e+10
    # R: ml145, full, L: ml146, full, A: ml147, full, B: ml148, full, y: ml149, full
    ml150 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml147, ml150, info) = LinearAlgebra.LAPACK.getrf!(ml147)

    # R: ml145, full, L: ml146, full, B: ml148, full, y: ml149, full, P11: ml150, ipiv, L9: ml147, lower_triangular_udiag, U10: ml147, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml147, ml148)

    # R: ml145, full, L: ml146, full, y: ml149, full, P11: ml150, ipiv, L9: ml147, lower_triangular_udiag, tmp53: ml148, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml147, ml148)

    # R: ml145, full, L: ml146, full, y: ml149, full, P11: ml150, ipiv, tmp54: ml148, full
    ml151 = [1:length(ml150);]
    @inbounds for i in 1:length(ml150)
        ml151[i], ml151[ml150[i]] = ml151[ml150[i]], ml151[i];
    end;
    ml152 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml152 = ml148[:,invperm(ml151)]

    # R: ml145, full, L: ml146, full, y: ml149, full, tmp55: ml152, full
    ml153 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml153, ml152)

    # R: ml145, full, L: ml146, full, y: ml149, full, tmp25: ml153, full
    ml154 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml153, 0.0, ml154)

    # R: ml145, full, L: ml146, full, y: ml149, full, tmp19: ml154, symmetric_lower_triangular
    ml155 = diag(ml146)
    ml156 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml145, 1, ml156, 1)
    # tmp29 = (L R)
    for i = 1:size(ml145, 2);
        view(ml145, :, i)[:] .*= ml155;
    end;        

    # R: ml156, full, y: ml149, full, tmp19: ml154, symmetric_lower_triangular, tmp29: ml145, full
    for i = 1:2000-1;
        view(ml154, i, i+1:2000)[:] = view(ml154, i+1:2000, i);
    end;
    ml157 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml154, 1, ml157, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml145, ml156, 1.0, ml154)

    # y: ml149, full, tmp19: ml157, full, tmp31: ml154, full
    ml158 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml154, ml158, info) = LinearAlgebra.LAPACK.getrf!(ml154)

    # y: ml149, full, tmp19: ml157, full, P35: ml158, ipiv, L33: ml154, lower_triangular_udiag, U34: ml154, upper_triangular
    ml159 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml157, ml149, 0.0, ml159)

    # P35: ml158, ipiv, L33: ml154, lower_triangular_udiag, U34: ml154, upper_triangular, tmp32: ml159, full
    ml160 = [1:length(ml158);]
    @inbounds for i in 1:length(ml158)
        ml160[i], ml160[ml158[i]] = ml160[ml158[i]], ml160[i];
    end;
    ml161 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml161 = ml159[ml160]

    # L33: ml154, lower_triangular_udiag, U34: ml154, upper_triangular, tmp40: ml161, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml154, ml161)

    # U34: ml154, upper_triangular, tmp41: ml161, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml154, ml161)

    # tmp17: ml161, full
    # x = tmp17

    finish = time_ns()
    GC.enable(true)
    return (tuple(ml161), (finish-start)*1e-9)
end