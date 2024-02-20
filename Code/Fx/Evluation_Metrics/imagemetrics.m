
%This function performs various quality metrics on the fused image
%to judge the quality. All of these values judge the spectral quality
%of the image except for the last metric. 


%M=resized multispectral image
%P=panchromatic image
%F=fused image

function sol=imagemetrics(M,P,F)
M=double(M);
P=double(P);
d=size(F, 3);
disp(' ');

%band correlation coefficients: best is close to 0.
ccM=bandCorrs(M);
ccF=bandCorrs(F);
ccD=(ccM-ccF).^2;
if d==3
    ccD=(ccD(1)+ccD(2)+ccD(4))/3;
    sol(1)= ccD^.5;
else 
    ccD=sum(ccD)/6;
    sol(1)=ccD^.5;
end

%ERGAS: best is close to 0. 
sol(2)=ERGAS(M, F);

%Qave: best is close to 1. 
sol(3)=QAVE(M,F);

%RASE: best is close to 0.
sol(4)=RASE(M,F);


%RMSE: best is close to 0.
sol(5)=RMSE(M,F);

%SAM: best is close to 0. 
sol(6)=SAM(M,F);

%SID: best is close to 0.
sol(7)=SID(M,F);

%Spatial: best is close to 1. 
sp=spatial(F, P);
if d==3
    sol(8)=sum(sp)/3;
else 
    sol(8)=sum(sp)/4;
end
