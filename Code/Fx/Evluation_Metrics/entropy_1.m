function [s]=entropy_1(image)
% function s=entropy(image)
% 
% calculates the byte-wise entropy of an image (2d signal)
%
% jon rogers
%
% entropy is the average info
% sum over number of elements in the alphabet of
%    probability*information

histo=hist(double(image(:)),256);%
probs=histo/sum(histo);
probs(probs==0) = []; 
I = -log2(probs);                      % wlg modify

innerproduct=probs.*I;
s=sum(innerproduct);

