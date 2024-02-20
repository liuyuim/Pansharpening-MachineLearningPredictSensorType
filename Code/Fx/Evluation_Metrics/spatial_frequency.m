function sf = spatial_frequency(img)
%获得灰度图像或者多光谱图像单个波段的空间频率
% 输入:
%     灰度图像或多光谱图像的单个波段
% 输出:
%    空间频率
% history:
%    creat by chry 2008.4.7
%    将函数名修改成spatial_frequency


if nargin~= 1
    error('请输入一个灰度图像或者多光谱图像单个波段.');
elseif size(img,3)~=1
    error('输入图像应为单波段图像.');
end

SizeR = size(img,1);%行数
SizeC = size(img,2);%列数
timg = double(img);
%计算行频率
rf=diff(timg');%控行计算从第二列起到最后一列减其前一列的值
rf=rf.^2;%差的平方
rf=sum(rf(:))/(SizeR*SizeC);%所有差平方之和除以像素的个数

%计算列频率
cf=diff(timg);%控列计算从第二行起到最后一列减其前一行的值
cf=cf.^2;%差的平方
cf=sum(cf(:))/(SizeR*SizeC);%所有差平方之和除以像素的个数

sf=sqrt(rf+cf);%根据空间行频率的平方和空间列频率平方来计算整体空间频率


