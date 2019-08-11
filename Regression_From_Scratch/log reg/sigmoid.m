function sig = sigmoid(x,w)
f=w*x';
sig=1/(1+exp(-f));
end