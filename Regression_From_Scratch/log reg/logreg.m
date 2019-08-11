clc;
clear;
s=importdata('credit.txt');
s2=importdata('testcredit.mat');
[par1 par2]=size(s);
figure(1)
for i=1:par1
 if s(i,3)==1
     plot(s(i,1),s(i,2),'-or')
     hold on;
 else
     plot(s(i,1),s(i,2),'-og')
     hold on;
 end
end
x=ones(par1,par2);
x(:,2:par2)=s(:,1:par2-1);
y=s(:,par2);
%% Grad Descent
 w=[0 0.05 -0.05];
topcount=1;
alph=.1;
lambda=3;
for lo=1:6000
    for i=1:par1
        f=1/(1+exp(-1*w*x(i,:)'));
        w=w-(alph*(f-y(i))*(x(i,:)))/par1; 
    end 
    w=w-alph*lambda*w/par1;
end
figure('Name',"Gradient Descent");
for i=1:par1
 if s(i,3)==1
     plot(s(i,1),s(i,2),'-or')
     hold on
 else
     plot(s(i,1),s(i,2),'-og')
     hold on
 end
end
xx=linspace(1.5,5,1000);
yy=(-w(2)*xx);
yy=yy+(-w(1));
yy=yy/w(3);
plot(xx,yy,'k')
title('Gradient Descent');
acc=0;

for i=1:par1
    f=1/(1+exp(-1*w*x(i,:)'));
    if  y(i)==round(f)
        acc=acc+1;
    end
end

acc
hold off
%% Newton Rhapson
w=[0 0.05 -0.05];
for lo=1:6000
        R=eye(par1);
        f=zeros(par1,1);
        J=0;
        for i=1:par1
           f(i,1)=1/(1+exp(-1*w*x(i,:)'));
           R(i,i)=f(i,1)*(1-f(i,1)); 
        end
        
        ki=[0;w(2:3)'];
        J=x'*(f-y);
        lambda=.01;
        J=(J+lambda*ki)/par1;
        
        
        ko=eye(3);
        ko(1,1)=0;
        H=x'*R*x;
        H=(H+lambda*ko)/par1;
        w=w-((H^-1)*J)';           
end

figure('Name','Newton Rhapson');
for i=1:par1
 if s(i,3)==1
     plot(s(i,1),s(i,2),'-or')
     hold on
 else
     plot(s(i,1),s(i,2),'-og')
     hold on
 end
end

xx=linspace(1.5,5,1000);
yy=(-w(2)*xx);
yy=yy+(-w(1));
yy=yy/w(3);
plot(xx,yy,'k')
title('Newton Rhapson');
acc2=0;

for i=1:par1
    f=1/(1+exp(-1*w*x(i,:)'));
    if  y(i)==round(f)
        acc2=acc2+1;
    end
end
acc2
 %% Transformed gradient descent
n=3
mom=x;
x=featuretransform(x(:,2:3),n);
w=randi(2,1,((n+1)*(n+2))/2)/10-.15;
alph=1;
lambda=10e-15;
for lo=1:10000
    for i=1:par1
           f=1/(1+exp(-1*w*x(i,:)'));
           w=w-(alph*(f-y(i))*(x(i,:)))/par1; 
        end
        w=w-alph*lambda*w/par1;
end
xx=linspace(0,7,1000);
yy=linspace(1,6,1000);
threshold=.01;
figure('Name','Transformed Gradient Descent');

for i=1:1000
 if s2.label(i)==1
     plot(s2.data(i,1),s2.data(i,2),'-or')
     hold on;
 else
     plot(s2.data(i,1),s2.data(i,2),'-og')
     hold on;
 end
end

for i=1:1000
    for j=1:1000
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
title('Transformed Gradient Descent');
acc3=0;

x=featuretransform(s2.data,n);
for i=1:1000
    f=sigmoid(x(i,:),w);
    [s2.label(i) round(f)];
    if  s2.label(i)==round(f)
        acc3=acc3+1;
    end
end
acc3=acc3/10
hold off
        
%% Transformed Newton Rhapson
x=mom;
x=featuretransform(x(:,2:3),n); 

not=((n+1)*(n+2))/2;
w=randi(2,1,not)/10-.15;
lambda=10^-10;
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
figure('Name','Transformed Newton Rhapson');
xx=linspace(1.5,5,1000);
yy=linspace(2,6,1000);
acc2=0;
threshold=.01;
for i=1:1000
    for j=1:1000
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
title('Transformed Newton Rhapson ');
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
    f=sigmoid(x(i,:),w);
    [s2.label(i) round(f)];
    if  s2.label(i)==round(f)
        acc2=acc2+1;
    end
end
acc2=acc2/10
hold off        
    





    




    




