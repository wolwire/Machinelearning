clc;
clear;
s=importdata('credit.txt');
s2=importdata('testcredit.mat');
[par1 par2]=size(s);
x=ones(par1,3);
x(:,2:3)=s(:,1:2);
y=s(:,3);

n=2;
 %% Newton Rhapson
x=featuretransform(x(:,2:3),n); 
x=x;
not=((n+1)*(n+2))/2;
w=randi(2,1,not)/10-.15;
lambda=10^-2;
f=0;
for lo=1:30
    J=zeros(not,1);
    H=zeros(not,not);
    for i=1:par1
           f=sigmoid(x(i,:),w);
           J=J+((f-y(i))*x(i,:)');
           H=H+(f*(1-f)*x(i,:)'*x(i,:));
    end
    ki=[0;w(2:not)'];
    J=(J+lambda*ki)/par1;
    ko=eye(not);
    ko(1,1)=0;
    H=(H+lambda*ko)/par1;
        w=w-((H^-1)*J)';
end

xx=linspace(1.5,5,300);
yy=linspace(2,6,300);
acc2=0;
threshold=.01;
for i=1:300
    for j=1:300
        p=[xx(i) yy(j)];
        p=featuretransform(p,n);
        p=p;
        f=sigmoid(p,w);
        if(abs(f-(0.5))<threshold)
            plot(xx(i),yy(j),'k.');
            hold on;
        end
    end
end
title('Gradient Descent');
for i=1:1000
 if s2.label(i)==1
     plot(s2.data(i,1),s2.data(i,2),'-or')
     hold on;
 else
     plot(s2.data(i,1),s2.data(i,2),'-og')
     hold on;
 end
end
x=featuretransform(s2.data,n);
for i=1:1000
    f=1/(1+exp(-1*w*x(i,:)'))
    if  s2.label(i)==round(f)
        acc2=acc2+1;
    end
end
acc2
hold off