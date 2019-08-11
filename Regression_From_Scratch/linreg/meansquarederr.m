function mse = meansquarederr(T, Tdash)
mse=0;
[m,n]=size(T);
for i=1:m
    mse=mse+(T(i)-Tdash(i))^2;
end
mse=mse/(m);
end
