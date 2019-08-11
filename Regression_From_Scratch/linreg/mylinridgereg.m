function beta = mylinridgereg(X,Y,lambda)
f=X'*X;
m=size(f);
I=eye(m);
beta =((f+lambda*I)^-1)*X'*Y;
end
