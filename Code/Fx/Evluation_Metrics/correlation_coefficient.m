function cc = correlation_coefficient(img1,img2)
%获得两个灰度图像或者多光谱图像单独波段间的相关系数
% 输入:
%     两幅单波段图像,输入顺序可以互换
% 输出:
%    相关系数
% history:
%      creat by wlg 2007.3.8
%      modify by chry 2008.4.8

if nargin~= 2
    error('请输入两幅图像.');
elseif size(img1,3)~=1 && size(img2,3)~=1
    error('输入图像应为单波段图像.');
end

I1 = double(img1);
I2 = double(img2);


%----计算波段像素的平均值
  temp = I1;
  av_I1 = mean(temp(:));
  I1 = I1 - av_I1;
  
  temp = I2;
  av_I2 = mean(temp(:));
  I2 = I2 - av_I2;
  
  t1 = sum(sum(I1.^2));
  t2 = sum(sum(I2.^2));
  cc = sum(sum(I1.* I2)) / sqrt(t1*t2);
