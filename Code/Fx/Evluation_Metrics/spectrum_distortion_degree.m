function sdd = spectrum_distortion_degree(img1,img2)
%获得两个灰度图像或者多光谱图像单独波段间的光谱扭曲的程度
%img1是原始图像img2是融合所得图像
% 输入:
%     两幅单波段图像,输入顺序可以互换
% 输出:
%    光谱扭曲度
% history:
%      creat by chry 2008.4.8

if nargin~= 2
    error('请输入两幅图像.');
elseif size(img1,3)~=1 && size(img2,3)~=1 %#ok<AND2>
    error('输入图像应为单波段图像.');
end
r1 = size(img1,1);%行数 
c1 = size(img1,2);
r2 = size(img2,1);
c2 = size(img2,2);%列数
if (r1~=r2 ||c1~=c2)
    error('请输入两幅同样大小的图像.');
end

timg1 = double(img1);
timg2 = double(img2);
timg=abs(timg1-timg2);
sdd=sum(sum(timg))/(r1*c1);
