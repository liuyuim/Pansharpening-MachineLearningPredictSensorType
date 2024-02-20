function cc = correlation_coefficient(img1,img2)
%��������Ҷ�ͼ����߶����ͼ�񵥶����μ�����ϵ��
% ����:
%     ����������ͼ��,����˳����Ի���
% ���:
%    ���ϵ��
% history:
%      creat by wlg 2007.3.8
%      modify by chry 2008.4.8

if nargin~= 2
    error('����������ͼ��.');
elseif size(img1,3)~=1 && size(img2,3)~=1
    error('����ͼ��ӦΪ������ͼ��.');
end

I1 = double(img1);
I2 = double(img2);


%----���㲨�����ص�ƽ��ֵ
  temp = I1;
  av_I1 = mean(temp(:));
  I1 = I1 - av_I1;
  
  temp = I2;
  av_I2 = mean(temp(:));
  I2 = I2 - av_I2;
  
  t1 = sum(sum(I1.^2));
  t2 = sum(sum(I2.^2));
  cc = sum(sum(I1.* I2)) / sqrt(t1*t2);
