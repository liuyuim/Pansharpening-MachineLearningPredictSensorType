function ce = cross_entropy(img1,img2)
%获得两个灰度图像或者多光谱图像单独波段间的交叉熵
%img1是原始图像img2是融合所得图像
% 输入:
%     两幅单波段图像,输入顺序可以互换
% 输出:
%    交叉熵
% history:
%      creat by chry 2008.4.8

if nargin~= 2
    error('请输入两幅图像.');
elseif size(img1,3)~=1 & size(img2,3)~=1
    error('输入图像应为单波段图像.');
end
timg1 = double(img1);
timg2 = double(img2);
%向量化
timg1=timg1(:);
timg2=timg2(:);
%求图像的灰度范围
tmin1=min(timg1);
tmin2=min(timg2);
tmax1=max(timg1);
tmax2=max(timg2);
tmin=min(tmin1,tmin2);
tmax=max(tmax1,tmax2);
%两幅图像的灰度分布概率
histo1=hist(timg1,tmin:tmax);
histo2=hist(timg2,tmin:tmax);
%去除两幅图像中在那些灰度值分布为0的影响
histo2(histo1==0)=1;
histo1(histo1==0)=1;
histo1(histo2==0)=1;
histo2(histo2==0)=1;
% histo1=histo1+eps;
% histo2=histo2+eps;
histo1=histo1/sum(histo1);
histo2=histo2/sum(histo2);
%求两幅图像在对应灰度值上的分布概率的商
histo=histo1./histo2;
histo=log2(histo);
histo=histo1.*histo;
ce=sum(histo);

