function sd = Standard_deviation(img)
%��ûҶ�ͼ����߶����ͼ�񵥸����εı�׼��
% ����:
%     �Ҷ�ͼ�������ͼ��ĵ�������
% ���:
%    ��׼��
% history:
%      creat by chry 2008.4.6


% if nargin~= 1
%     error('������һ���Ҷ�ͼ����߶����ͼ�񵥸�����.');
% elseif size(img,3)~=1
%     error('����ͼ��ӦΪ������ͼ��.');
% end
% SizeR = size(img,1);%����
% SizeC = size(img,2);%����
% timg = double(img);
% timg=timg(:);
% av_img= mean(timg(:));%----���㲨�����ص�ƽ��ֵ
% d_img = timg  - av_img;% ����ÿ�����ص�ƫ��
% sd_img=d_img.^2;%����ƫ���ƽ��
% nf=sum(sd_img);%��������ƫ��ĺ�
% nf=nf/(SizeR*SizeC-1);%ƫ��ĺͳ������صĸ�����1
% nf=sqrt(nf);%��ƽ��
%��ֱ�ӵ������еĺ���std2
sd=std2(img);

  
  
