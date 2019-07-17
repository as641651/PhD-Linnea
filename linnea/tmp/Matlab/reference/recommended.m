function [res, time] = recommended(R, L, A, B, y)
    tic;
    x = ((((((transpose(A))\transpose(B))*B/(A))+transpose(R)*L*R))\((transpose(A))\transpose(B)))*B*((A)\y);    
    time = toc;
    res = {x};
end