clc;
clear;
s=importdata('linregdata');
data_size=.2;
[m,n]=size(s.data);
x=zeros(m,n+4);
x(:,5:n+4)=s.data(:,1:n);
x(:,1)=ones(m,1);
%Part 1
for i =1:m
    if strcmp(s.textdata(i),'M')
        x(i,2)=1;
    elseif strcmp(s.textdata(i),'F')
        x(i,3)=1;
    else 
        x(i,4)=1;
    end
end

lambda=5;
frac=0.70;
f=0;
%Part 4 
for l=1:100
%k=importdata('random_no.txt');
k=randperm(m);
p1=floor(data_size*m);
p2=floor(m*frac)+p1;
train_x=x(k(p1+1:p2),1:n+3);
train_y=x(k(p1+1:p2),n+4);
siz_train=size(train_x);
%Part4 (validation)
valid_x=x(k(p2+1:m),1:n+3);
valid_y=x(k(p2+1:m),n+4);
siz_valid=size(valid_x);
%Part4 (test)
test_x=x(k(1:p1),1:n+3);
test_y=x(k(1:p1),n+4);
siz_test=size(test_x);
%Part 2 (Standardization)
train_avg=mean(train_x);
train_stddev=std(train_x);
test_avg=mean(test_x);
test_stddev=std(test_x);
valid_avg=mean(valid_x);
valid_stddev=std(valid_x);
for i=2:n+3
 train_x(:,i)=(train_x(:,i)-train_avg(i)*ones(siz_train(1),1))/train_stddev(i);
 test_x(:,i)=(test_x(:,i)-train_avg(i)*ones(siz_test(1),1))/train_stddev(i);
end
pom=size(train_y);
p=zeros(pom(1),2);
wts=mylinridgereg(train_x,train_y,lambda);
p(:,1)=mylinridgeregeval(train_x,wts);
p(:,2)=train_y(:);
msqe=meansquarederr(p(:,1),p(:,2));
f=f+msqe;
end
f/100;







        
