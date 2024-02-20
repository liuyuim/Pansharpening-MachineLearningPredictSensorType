function sf = spatial_frequency(img)
%��ûҶ�ͼ����߶����ͼ�񵥸����εĿռ�Ƶ��
% ����:
%     �Ҷ�ͼ�������ͼ��ĵ�������
% ���:
%    �ռ�Ƶ��
% history:
%    creat by chry 2008.4.7
%    ���������޸ĳ�spatial_frequency


if nargin~= 1
    error('������һ���Ҷ�ͼ����߶����ͼ�񵥸�����.');
elseif size(img,3)~=1
    error('����ͼ��ӦΪ������ͼ��.');
end

SizeR = size(img,1);%����
SizeC = size(img,2);%����
timg = double(img);
%������Ƶ��
rf=diff(timg');%���м���ӵڶ��������һ�м���ǰһ�е�ֵ
rf=rf.^2;%���ƽ��
rf=sum(rf(:))/(SizeR*SizeC);%���в�ƽ��֮�ͳ������صĸ���

%������Ƶ��
cf=diff(timg);%���м���ӵڶ��������һ�м���ǰһ�е�ֵ
cf=cf.^2;%���ƽ��
cf=sum(cf(:))/(SizeR*SizeC);%���в�ƽ��֮�ͳ������صĸ���

sf=sqrt(rf+cf);%���ݿռ���Ƶ�ʵ�ƽ���Ϳռ���Ƶ��ƽ������������ռ�Ƶ��


