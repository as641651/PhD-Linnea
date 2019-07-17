using LinearAlgebra.BLAS
using LinearAlgebra

function algorithm60(ml2008::Array{Float64,2}, ml2009::Array{Float64,2}, ml2010::Array{Float64,2}, ml2011::Array{Float64,2}, ml2012::Array{Float64,1})
    # cost 5.07e+10
    # R: ml2008, full, L: ml2009, full, A: ml2010, full, B: ml2011, full, y: ml2012, full
    ml2013 = Array{Float64}(undef, 2000, 2000)
    # tmp26 = B^T
    transpose!(ml2013, ml2011)

    # R: ml2008, full, L: ml2009, full, A: ml2010, full, y: ml2012, full, tmp26: ml2013, full
    ml2014 = Array{Float64}(undef, 2000)
    # (P11^T L9 U10) = A
    (ml2010, ml2014, info) = LinearAlgebra.LAPACK.getrf!(ml2010)

    # R: ml2008, full, L: ml2009, full, y: ml2012, full, tmp26: ml2013, full, P11: ml2014, ipiv, L9: ml2010, lower_triangular_udiag, U10: ml2010, upper_triangular
    # tmp27 = (U10^-T tmp26)
    trsm!('L', 'U', 'T', 'N', 1.0, ml2010, ml2013)

    # R: ml2008, full, L: ml2009, full, y: ml2012, full, P11: ml2014, ipiv, L9: ml2010, lower_triangular_udiag, tmp27: ml2013, full
    # tmp28 = (L9^-T tmp27)
    trsm!('L', 'L', 'T', 'U', 1.0, ml2010, ml2013)

    # R: ml2008, full, L: ml2009, full, y: ml2012, full, P11: ml2014, ipiv, tmp28: ml2013, full
    ml2015 = [1:length(ml2014);]
    @inbounds for i in 1:length(ml2014)
        ml2015[i], ml2015[ml2014[i]] = ml2015[ml2014[i]], ml2015[i];
    end;
    ml2016 = Array{Float64}(undef, 2000, 2000)
    # tmp25 = (P11^T tmp28)
    ml2016 = ml2013[invperm(ml2015),:]

    # R: ml2008, full, L: ml2009, full, y: ml2012, full, tmp25: ml2016, full
    ml2017 = Array{Float64}(undef, 2000, 2000)
    # tmp19 = (tmp25 tmp25^T)
    syrk!('L', 'N', 1.0, ml2016, 0.0, ml2017)

    # R: ml2008, full, L: ml2009, full, y: ml2012, full, tmp19: ml2017, symmetric_lower_triangular
    ml2018 = Array{Float64}(undef, 2000)
    # tmp32 = (tmp19 y)
    symv!('L', 1.0, ml2017, ml2012, 0.0, ml2018)

    # R: ml2008, full, L: ml2009, full, tmp19: ml2017, symmetric_lower_triangular, tmp32: ml2018, full
    ml2019 = diag(ml2009)
    ml2020 = Array{Float64}(undef, 1999, 2000)
    blascopy!(1999*2000, ml2008, 1, ml2020, 1)
    # tmp29 = (L R)
    for i = 1:size(ml2008, 2);
        view(ml2008, :, i)[:] .*= ml2019;
    end;        

    # R: ml2020, full, tmp19: ml2017, symmetric_lower_triangular, tmp32: ml2018, full, tmp29: ml2008, full
    for i = 1:2000-1;
        view(ml2017, i, i+1:2000)[:] = view(ml2017, i+1:2000, i);
    end;
    # tmp31 = (tmp19 + (R^T tmp29))
    gemm!('T', 'N', 1.0, ml2020, ml2008, 1.0, ml2017)

    # tmp32: ml2018, full, tmp31: ml2017, full
    ml2021 = Array{Float64}(undef, 2000)
    # (P35^T L33 U34) = tmp31
    (ml2017, ml2021, info) = LinearAlgebra.LAPACK.getrf!(ml2017)

    # tmp32: ml2018, full, P35: ml2021, ipiv, L33: ml2017, lower_triangular_udiag, U34: ml2017, upper_triangular
    ml2022 = [1:length(ml2021);]
    @inbounds for i in 1:length(ml2021)
        ml2022[i], ml2022[ml2021[i]] = ml2022[ml2021[i]], ml2022[i];
    end;
    ml2023 = Array{Float64}(undef, 2000)
    # tmp40 = (P35 tmp32)
    ml2023 = ml2018[ml2022]

    # L33: ml2017, lower_triangular_udiag, U34: ml2017, upper_triangular, tmp40: ml2023, full
    # tmp41 = (L33^-1 tmp40)
    trsv!('L', 'N', 'U', ml2017, ml2023)

    # U34: ml2017, upper_triangular, tmp41: ml2023, full
    # tmp17 = (U34^-1 tmp41)
    trsv!('U', 'N', 'N', ml2017, ml2023)

    # tmp17: ml2023, full
    # x = tmp17
    return (ml2023)
end