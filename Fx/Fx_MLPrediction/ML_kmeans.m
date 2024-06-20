% MATLAB MachineLearning/kmeans/kmeans20181218_2D.m
% https://www.bilibili.com/video/BV15t411v7Uj/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5
% https://github.com/CodeSama346623/Bilibili346623/blob/master/MATLAB%20MachineLearning/kmeans/kmeans20181218_2D.m
% https://github.com/CodeSama346623/Bilibili346623/blob/master/MATLAB%20MachineLearning/kmeans/kmeans20181218_3D.m
% %% 用球体/正方体/锥体

function [] = ML_kmeans(EvaluationDirList,TrainProportion,columnSample)

%% 初始化工作空间
% clc
% clear
% close all

%% 载入数据
% 分特征数据和标签数据,分训练集测试集
[features,label,train_id,test_id,index_id] = MLMatrixRead(EvaluationDirList,TrainProportion,columnSample);
    

%% 三维数据
% 花瓣长度、花瓣宽度、花萼长度散点图(无标记)
species = label;

figure;
speciesNum = grp2idx(species);
plot3(features(:,index_id(1)),features(:,index_id(2)),features(:,index_id(3)),'.','MarkerSize',20)
grid on
view([60,24])
xlabel('Pansharpened In GF1 Model')
ylabel('Pansharpened In QB Model')
zlabel('Pansharpened In WV4 Model')
title('无标记')
set(gca,'FontSize',12)
set(gca,'FontWeight','Bold')

% 花瓣长度、花瓣宽度、花萼长度散点图(真实标记)
figure;
hold on
colorArray=['r','g','b'];
for  i= 1:3
    plotind=speciesNum==i;
    plot3(features(plotind,index_id(1)),features(plotind,index_id(2)),features(plotind,index_id(3)),'.','MarkerSize',20,'MarkerEdgeColor',colorArray(i))
end
hold off
grid on
view([60,24])
xlabel('Pansharpening In GF1 Model')
ylabel('Pansharpening In QB Model')
zlabel('Pansharpening In WV4 Model')
title('真实标记')
set(gca,'FontSize',12)
set(gca,'FontWeight','Bold')

% 添加图例
legend('GF1 Image', 'QB Image', 'WV4 Image')

%% kmeans 聚类
data2=[ features(:,index_id(1)), features(:,index_id(2)),features(:,index_id(3))];
K=3;
[idx2,cen2]=kmeans(data2,K,'Distance','sqeuclidean','Replicates',5,'Display','Final');
% 调整标号
dist2=sum(cen2.^2,2);
[dump2,sortind2]=sort(dist2,'ascend');
newidx2=zeros(size(idx2));
for i =1:K
    newidx2(idx2==i)=find(sortind2==i);
end

% 花瓣长度和花瓣宽度散点图(真实标记:实心圆+kmeans分类:圈)
figure;
hold on
colorArray=['r','g','b'];
for  i= 1:3
    plotind=speciesNum==i;
    plot3(features(plotind,index_id(1)),features(plotind,index_id(2)),features(plotind,index_id(3)),'.','MarkerSize',15,'MarkerEdgeColor',colorArray(i))
end

for  i= 1:3
    plotind=newidx2==i;
    plot3(features(plotind,index_id(1)),features(plotind,index_id(2)),features(plotind,index_id(3)),'o','MarkerSize',10,'MarkerEdgeColor',colorArray(i))
end
for i=1:3
    plot3(cen2(i,1),cen2(i,2),cen2(i,3),'*m')
end

hold off
grid on
view([60,24])
xlabel('Pansharpening In GF1 Model')
ylabel('Pansharpening In QB Model')
zlabel('Pansharpening In WV4 Model')
title('真实标记:实心圆+kmeans分类:圈')
set(gca,'FontSize',12)
set(gca,'FontWeight','Bold')

%% 混淆矩阵 ConfusionMatrix
confMat=confusionmat(speciesNum,newidx2)
error23=speciesNum==2&newidx2==3;
errDat23=data2(error23,:)
error32=speciesNum==3&newidx2==2;
errDat32=data2(error32,:)
[dump, errdatSort]=sort(errDat32(:,3));
errDat32Sort=errDat32(errdatSort,:)