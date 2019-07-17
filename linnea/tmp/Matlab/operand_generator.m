function [out] = operand_generator()
    import MatrixGenerator.*;
    out{ 1 } = generate([1999,2000], Shape.UpperTriangular(), Properties.Random([10, 11]));
    out{ 2 } = generate([1999,1999], Shape.Symmetric(), Shape.Diagonal(), Shape.LowerTriangular(), Shape.UpperTriangular(), Properties.Random([10, 11]));
    out{ 3 } = generate([2000,2000], Shape.General(), Properties.Random([-1, 1]));
    out{ 4 } = generate([2000,2000], Shape.General(), Properties.Random([-1, 1]));
    out{ 5 } = generate([2000,1], Shape.General(), Properties.Random([-1, 1]));
end