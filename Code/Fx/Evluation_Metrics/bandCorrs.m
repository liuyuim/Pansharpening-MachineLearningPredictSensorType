function bandCorrs=bandCorrs(M)
M=double(M);
[n,m,d]=size(M);
band12=corrcoef(M(:,:,1), M(:,:,2));
band13=corrcoef(M(:,:,1), M(:,:,3));
if (d==4)
    band14=corrcoef(M(:,:,1), M(:,:,4));
    band24=corrcoef(M(:,:,2), M(:,:,4));
    band34=corrcoef(M(:,:,3), M(:,:,4));
else
    band14=zeros(2);
    band24=zeros(2);
    band34=zeros(2);
end
band23=corrcoef(M(:,:,2), M(:,:,3));


bandCorrs=[band12(1,2), band13(1,2), band14(1,2), band23(1,2), band24(1,2), band34(1,2)];