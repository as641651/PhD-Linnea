using MatrixGenerator

function operand_generator()
    R::UpperTriangular{Float64,Array{Float64,2}} = generate((1999,2000), [Shape.UpperTriangular, Properties.Random(10, 11)])
    L::Diagonal{Float64,Array{Float64,1}} = generate((1999,1999), [Shape.Symmetric, Shape.Diagonal, Shape.LowerTriangular, Shape.UpperTriangular, Properties.Random(10, 11)])
    A::Array{Float64,2} = generate((2000,2000), [Shape.General, Properties.Random(-1, 1)])
    B::Array{Float64,2} = generate((2000,2000), [Shape.General, Properties.Random(-1, 1)])
    y::Array{Float64,1} = generate((2000,1), [Shape.General, Properties.Random(-1, 1)])
    return (R, L, A, B, y,)
end