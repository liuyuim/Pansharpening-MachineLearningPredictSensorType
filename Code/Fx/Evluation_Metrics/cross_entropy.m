function ce = cross_entropy(img1,img2)
%��������Ҷ�ͼ����߶����ͼ�񵥶����μ�Ľ�����
%img1��ԭʼͼ��img2���ں�����ͼ��
% ����:
%     ����������ͼ��,����˳����Ի���
% ���:
%    ������
% history:
%      creat by chry 2008.4.8

if nargin~= 2
    error('����������ͼ��.');
elseif size(img1,3)~=1 & size(img2,3)~=1
    error('����ͼ��ӦΪ������ͼ��.');
end
timg1 = double(img1);
timg2 = double(img2);
%������
timg1=timg1(:);
timg2=timg2(:);
%��ͼ��ĻҶȷ�Χ
tmin1=min(timg1);
tmin2=min(timg2);
tmax1=max(timg1);
tmax2=max(timg2);
tmin=min(tmin1,tmin2);
tmax=max(tmax1,tmax2);
%����ͼ��ĻҶȷֲ�����
histo1=hist(timg1,tmin:tmax);
histo2=hist(timg2,tmin:tmax);
%ȥ������ͼ��������Щ�Ҷ�ֵ�ֲ�Ϊ0��Ӱ��
histo2(histo1==0)=1;
histo1(histo1==0)=1;
histo1(histo2==0)=1;
histo2(histo2==0)=1;
% histo1=histo1+eps;
% histo2=histo2+eps;
histo1=histo1/sum(histo1);
histo2=histo2/sum(histo2);
%������ͼ���ڶ�Ӧ�Ҷ�ֵ�ϵķֲ����ʵ���
histo=histo1./histo2;
histo=log2(histo);
histo=histo1.*histo;
ce=sum(histo);

