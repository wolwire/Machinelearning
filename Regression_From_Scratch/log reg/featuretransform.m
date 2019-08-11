function features = featuretransform(X,degree)
[m n]=size(X);
features=zeros(m,(n+1)*(n+2)/2);
size(features);
lol=1;
for i=1:degree+1
    for j=1:degree+1
        if (i+j<degree+3)
            features(:,lol)=X(:,1).^(i-1).*X(:,2).^(j-1);
            lol=lol+1;
        end
    end
end
end

            
            