using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm57(ml1906::Array{Float64,2}, ml1907::Array{Float64,2}, ml1908::Array{Float64,2}, ml1909::Array{Float64,2}, ml1910::Array{Float64,1})
    # cost 5.07e+10
    # R: ml1906, full, L: ml1907, full, A: ml1908, full, B: ml1909, full, y: ml1910, full
    ml1911 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml1908, ml1911, info) = LinearAlgebra.LAPACK.getrf!(ml1908)

    # R: ml1906, full, L: ml1907, full, B: ml1909, full, y: ml1910, full, P11: ml1911, ipiv, L9: ml1908, lower_triangular_udiag, U10: ml1908, upper_triangular
    # tmp53 = (B U10^-1)
    trsm!('R', 'U', 'N', 'N', 1.0, ml1908, ml1909)

    # R: ml1906, full, L: ml1907, full, y: ml1910, full, P11: ml1911, ipiv, L9: ml1908, lower_triangular_udiag, tmp53: ml1909, full
    # tmp54 = (tmp53 L9^-1)
    trsm!('R', 'L', 'N', 'U', 1.0, ml1908, ml1909)

    # R: ml1906, full, L: ml1907, full, y: ml1910, full, P11: ml1911, ipiv, tmp54: ml1909, full
    ml1912 = [1:length(ml1911);]
    @inbounds for i in 1:length(ml1911)
        ml1912[i], ml1912[ml1911[i]] = ml1912[ml1911[i]], ml1912[i];
    end;
    ml1913 = Array{Float64}(undef, 2000, 2000)
    # tmp55 = (tmp54 P11)
    ml1913 = ml1909[:,invperm(ml1912)]

    # R: ml1906, full, L: ml1907, full, y: ml1910, full, tmp55: ml1913, full
    ml1914 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = tmp55^T
    transpose!(ml1914, ml1913)

    # R: ml1906, full, L: ml1907, full, y: ml1910, full, tmp25: ml1914, full
    ml1915 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml1914, 0.0, ml1915)

    # R: ml1906, full, L: ml1907, full, y: ml1910, full, tmp19: ml1915, symmetric_lower_triangular
    ml1916 = diag(ml1907)
    ml1917 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml1906, 1, ml1917, 1)
    # tmp29 = (L R)
    for i = 1:size(ml1906, 2);
        view(ml1906, :, i)[:] .*= ml1916;
    end;        

    # R: ml1917, full, y: ml1910, full, tmp19: ml1915, symmetric_lower_triangular, tmp29: ml1906, full
    for i = 1:2000-1;
        view(ml1915, i, i+1:2000)[:] = view(ml1915, i+1:2000, i);
    end;
    ml1918 = Array{Float64}(undef, 2000, 2000)
    blascopy!(2000*2000, ml1915, 1, ml1918, 1)
    # tmp31 = (tmp19 + (tmp29^T R))
    gemm!('T', 'N', 1.0, ml1906, ml1917, 1.0, ml1915)

    # y: ml1910, full, tmp19: ml1918, full, tmp31: ml1915, full
    ml1919 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml1915, ml1919, info) = LinearAlgebra.LAPACK.getrf!(ml1915)

    # y: ml1910, full, tmp19: ml1918, full, P35: ml1919, ipiv, L33: ml1915, lower_triangular_udiag, U34: ml1915, upper_triangular
    ml1920 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml1918, ml1910, 0.0, ml1920)

    # P35: ml1919, ipiv, L33: ml1915, lower_triangular_udiag, U34: ml1915, upper_triangular, tmp32: ml1920, full
    ml1921 = [1:length(ml1919);]
    @inbounds for i in 1:length(ml1919)
        ml1921[i], ml1921[ml1919[i]] = ml1921[ml1919[i]], ml1921[i];
    end;
    ml1922 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml1922 = ml1920[ml1921]

    # L33: ml1915, lower_triangular_udiag, U34: ml1915, upper_triangular, tmp40: ml1922, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml1915, ml1922)

    # U34: ml1915, upper_triangular, tmp41: ml1922, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml1915, ml1922)

    # tmp17: ml1922, full
    # x = tmp17
    return (ml1922)
end