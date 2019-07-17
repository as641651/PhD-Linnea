using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm27(ml902::Array{Float64,2}, ml903::Array{Float64,2}, ml904::Array{Float64,2}, ml905::Array{Float64,2}, ml906::Array{Float64,1})
    # cost 5.07e+10
    # R: ml902, full, L: ml903, full, A: ml904, full, B: ml905, full, y: ml906, full
    ml907 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml907, ml905)

    # R: ml902, full, L: ml903, full, A: ml904, full, y: ml906, full, tmp26: ml907, full
    ml908 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml904, ml908, info) = LinearAlgebra.LAPACK.getrf!(ml904)

    # R: ml902, full, L: ml903, full, y: ml906, full, tmp26: ml907, full, P11: ml908, ipiv, L9: ml904, lower_triangular_udiag, U10: ml904, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml904, ml907)

    # R: ml902, full, L: ml903, full, y: ml906, full, P11: ml908, ipiv, L9: ml904, lower_triangular_udiag, tmp27: ml907, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml904, ml907)

    # R: ml902, full, L: ml903, full, y: ml906, full, P11: ml908, ipiv, tmp28: ml907, full
    ml909 = [1:length(ml908);]
    @inbounds for i in 1:length(ml908)
        ml909[i], ml909[ml908[i]] = ml909[ml908[i]], ml909[i];
    end;
    ml910 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml910 = ml907[invperm(ml909),:]

    # R: ml902, full, L: ml903, full, y: ml906, full, tmp25: ml910, full
    ml911 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml910, 0.0, ml911)

    # R: ml902, full, L: ml903, full, y: ml906, full, tmp19: ml911, symmetric_lower_triangular
    ml912 = diag(ml903)
    ml913 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml902, 1, ml913, 1)
    # tmp29 = (L R)
    for i = 1:size(ml902, 2);
        view(ml902, :, i)[:] .*= ml912;
    end;        

    # R: ml913, full, y: ml906, full, tmp19: ml911, symmetric_lower_triangular, tmp29: ml902, full
    for i = 1:2000-1;
        view(ml911, i, i+1:2000)[:] = view(ml911, i+1:2000, i);
    end;
    ml914 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml911, 1, ml914, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml902, ml913, 1.0, ml911)

    # y: ml906, full, tmp19: ml914, full, tmp31: ml911, full
    ml915 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml911, ml915, info) = LinearAlgebra.LAPACK.getrf!(ml911)

    # y: ml906, full, tmp19: ml914, full, P35: ml915, ipiv, L33: ml911, lower_triangular_udiag, U34: ml911, upper_triangular
    ml916 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml914, ml906, 0.0, ml916)

    # P35: ml915, ipiv, L33: ml911, lower_triangular_udiag, U34: ml911, upper_triangular, tmp32: ml916, full
    ml917 = [1:length(ml915);]
    @inbounds for i in 1:length(ml915)
        ml917[i], ml917[ml915[i]] = ml917[ml915[i]], ml917[i];
    end;
    ml918 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml918 = ml916[ml917]

    # L33: ml911, lower_triangular_udiag, U34: ml911, upper_triangular, tmp40: ml918, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml911, ml918)

    # U34: ml911, upper_triangular, tmp41: ml918, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml911, ml918)

    # tmp17: ml918, full
    # x = tmp17
    return (ml918)
end