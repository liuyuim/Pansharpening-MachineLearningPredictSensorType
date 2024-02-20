function sd = Standard_deviation(img)
%获得灰度图像或者多光谱图像单个波段的标准差
% 输入:
%     灰度图像或多光谱图像的单个波段
% 输出:
%    标准差
% history:
%      creat by chry 2008.4.6


% if nargin~= 1
%     error('请输入一个灰度图像或者多光谱图像单个波段.');
% elseif size(img,3)~=1
%     error('输入图像应为单波段图像.');
% end
% SizeR = size(img,1);%行数
% SizeC = size(img,2);%列数
% timg = double(img);
% timg=timg(:);
% av_img= mean(timg(:));%----计算波段像素的平均值
% d_img = timg  - av_img;% 计算每个像素的偏差
% sd_img=d_img.^2;%计算偏差的平方
% nf=sum(sd_img);%所有像素偏差的和
% nf=nf/(SizeR*SizeC-1);%偏差的和除以像素的个数减1
% nf=sqrt(nf);%开平方
%或直接调用已有的函数std2
sd=std2(img);

  
  
