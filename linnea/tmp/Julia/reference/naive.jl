using LinearAlgebra.BLAS
using LinearAlgebra

function naive(R::UpperTriangular{Float64,Array{Float64,2}}, L::Diagonal{Float64,Array{Float64,1}}, A::Array{Float64,2}, B::Array{Float64,2}, y::Array{Float64,1})
    start::Float64 = 0.0
    finish::Float64 = 0.0
    Benchmarker.cachescrub()
    GC.gc()
    GC.enable(false)
    start = time_ns()

    x = inv((transpose(R)*L*R+inv(transpose(A))*transpose(B)*B*inv(A)))*inv(transpose(A))*transpose(B)*B*inv(A)*y;

    finish = time_ns()
    GC.enable(true)
    return (tuple(x), (finish-start)*1e-9)
end