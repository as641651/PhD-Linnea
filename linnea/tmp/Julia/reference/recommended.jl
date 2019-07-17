using LinearAlgebra.BLAS
using LinearAlgebra

function recommended(R::UpperTriangular{Float64,Array{Float64,2}}, L::Diagonal{Float64,Array{Float64,1}}, A::Array{Float64,2}, B::Array{Float64,2}, y::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    x = ((((((transpose(A))\transpose(B))*B/(A))+transpose(R)*L*R))\((transpose(A))\transpose(B)))*B*((A)\y);

    finish = time_ns()
    GC.enable(true)
    return (tuple(x), (finish-start)*1e-9)
end