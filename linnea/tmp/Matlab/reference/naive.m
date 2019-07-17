function [res, time] = naive(R, L, A, B, y)
    tic;
    x = inv((transpose(R)*L*R+inv(transpose(A))*transpose(B)*B*inv(A)))*inv(transpose(A))*transpose(B)*B*inv(A)*y;    
    time = toc;
    res = {x};
end