function sdd = spectrum_distortion_degree(img1,img2)
%��������Ҷ�ͼ����߶����ͼ�񵥶����μ�Ĺ���Ť���ĳ̶�
%img1��ԭʼͼ��img2���ں�����ͼ��
% ����:
%     ����������ͼ��,����˳����Ի���
% ���:
%    ����Ť����
% history:
%      creat by chry 2008.4.8

if nargin~= 2
    error('����������ͼ��.');
elseif size(img1,3)~=1 && size(img2,3)~=1 %#ok<AND2>
    error('����ͼ��ӦΪ������ͼ��.');
end
r1 = size(img1,1);%���� 
c1 = size(img1,2);
r2 = size(img2,1);
c2 = size(img2,2);%����
if (r1~=r2 ||c1~=c2)
    error('����������ͬ����С��ͼ��.');
end

timg1 = double(img1);
timg2 = double(img2);
timg=abs(timg1-timg2);
sdd=sum(sum(timg))/(r1*c1);
