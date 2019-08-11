clc;
clear;
s=importdata('linregdata');
data_size=.2;
[m,n]=size(s.data);
x=zeros(m,n+4);
x(:,5:n+4)=s.data(:,1:n);
x(:,1)=ones(m,1);
%% Part 1
for i =1:m
    if strcmp(s.textdata(i),'M')
        x(i,2)=1;
    elseif strcmp(s.textdata(i),'F')
        x(i,3)=1;
    else 
        x(i,4)=1;
    end
end

frac=0.70;
msqe=zeros(100,100);
msqe2=zeros(100,100);

% Part 4 
for po=1:100
      k=randperm(m);
      po
for j=1:100
          
frac=(0.80-0.019)*j/100+0.019;
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
% Part 2 (Standardization)
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

% train model 
for qp=1:100 
    lambda=qp/10;
    wts=mylinridgereg(train_x,train_y,lambda);
    pom=size(train_y);
    p=zeros(pom(1),2);
    p(:,1)=mylinridgeregeval(train_x,wts);
    p(:,2)=train_y(:);
    msqe(j,qp)= msqe(j,qp)+meansquarederr(p(:,1),p(:,2));
    pom=size(test_y);
    p2=zeros(pom(1),2);
    p2(:,1)=mylinridgeregeval(test_x,wts);
    p2(:,2)=test_y(:);
    msqe2(j,qp)=msqe2(j,qp)+meansquarederr(p2(:,1),p2(:,2));
    end  
    end
end
msqe=msqe/100;
msqe2=msqe2/100;
for i=1:4
figure('Name','Train and test error(Part 7)');
plot((1:100)/10,msqe2(i*25,:),'r');
xlabel("Lambda");
ylabel("Mean Squared error");
hold on;
title('Test Error');
plot((1:100)/10,msqe(i*25,:),'b');
axis([0 10 4.5 6])
hold off;
end
figure('Name','test error');
plot((0.80-0.019)*(1:100)/100+0.019,min(msqe2,[],2),'r');

figure('Name','Train Min vs lambda')
[v d]=min(msqe2,[],2);
for l=1:100
minval=((0.80-0.019)*l/100+0.019);  
plot(d(l)/10,minval,'bl.')
hold on;
end
figure(4)
p2(:,1)=mylinridgeregeval(test_x,wts);
p2(:,2)=test_y(:);
scatter(p2(:,1),p2(:,2));
hold on;
sam=linspace(0,22,1000);
plot(sam,sam);
hold off;