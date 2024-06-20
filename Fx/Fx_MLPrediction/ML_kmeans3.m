% MATLAB MachineLearning/kmeans/kmeans20181218_2D.m
% https://www.bilibili.com/video/BV15t411v7Uj/?spm_id_from=333.337.search-card.all.click&vd_source=eb9a5338e2b00e3e19c0a29e3a5acfa5
% https://github.com/CodeSama346623/Bilibili346623/blob/master/MATLAB%20MachineLearning/kmeans/kmeans20181218_2D.m
% https://github.com/CodeSama346623/Bilibili346623/blob/master/MATLAB%20MachineLearning/kmeans/kmeans20181218_3D.m
% %%
% 在 MATLAB 的标准绘图环境中，创建真正的立体图形（如三维球体、锥体和立方体）比较复杂，因为 plot3 函数主要用于绘制点、线和标记，而不是用于创建复杂的三维形状。但是，您可以使用一些特定的函数来近似这些形状。例如：
% 
% 球体：可以使用 sphere 函数生成。
% 锥体：可以通过结合 cylinder 和 surf 函数生成。
% 立方体：可以通过 patch 或 surf 函数生成。
% 以下是一个示例，展示如何在同一图中绘制这三种形状：
% 
% matlab
% Copy code
% figure;
% hold on;
% 
% % 球体
% [x, y, z] = sphere;
% x = x * 0.1; % 调整大小
% y = y * 0.1;
% z = z * 0.1;
% surf(x, y, z, 'FaceColor', 'r');
% 
% % 锥体
% [t, z] = cylinder([0, 1]);
% z(2, :) = 0.1; % 调整高度
% surf(t, z, t, 'FaceColor', 'g');
% 
% % 立方体
% v = [-0.1 -0.1 -0.1; 0.1 -0.1 -0.1; 0.1 0.1 -0.1; -0.1 0.1 -0.1; -0.1 -0.1 0.1; 0.1 -0.1 0.1; 0.1 0.1 0.1; -0.1 0.1 0.1];
% f = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
% patch('Vertices', v, 'Faces', f, 'FaceColor', 'b');
% 
% axis equal;
% grid on;
% view([60, 24]);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% title('立体形状示例');
% 
% hold off;
% 这段代码将创建一个包含红色球体、绿色锥体和蓝色立方体的三维场景。请注意，这些形状的位置和大小可能需要根据您的具体需求进行调整。此外，创建这些形状的方式比较基础，可能无法完全满足所有视觉效果的需求。

% 在MATLAB中，可以使用不同的字符串来指定各种标记样式。对于散点图（如 plot3 函数），以下是一些常用的标记选项：
% 
% 'o'：圆圈
% '+'：加号
% '*'：星号
% '.'：点
% 'x'：叉号
% 's'：正方形
% 'd'：菱形
% '^'：向上的三角形（锥体）
% 'v'：向下的三角形
% '>'：向右的三角形
% '<'：向左的三角形
% 'p'：五角星
% 'h'：六角星
% 这些标记可以与颜色代码（如 'r' 表示红色）结合使用。例如，'ro' 会创建红色圆圈标记。需要注意的是，这些标记都是平面的，MATLAB 并不支持直接绘制立体的球体、立方体等图形作为散点图的标记。如果你需要更复杂的三维形状，可能需要使用其他绘图函数，如 scatter3 结合自定义的图形绘制方法。




%%

function [] = ML_kmeans3(EvaluationDirList,TrainProportion,columnSample)

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
markerArray = ['o', 's', '^']; % 定义标记数组
projectionMarkerArray  = ['o', 's', '^']; % 定义标记数组
markerSize = 5; % 定义标记的大小
for  i= 1:3
    plotind=speciesNum==i;
    plot3(features(plotind,index_id(1)),features(plotind,index_id(2)),features(plotind,index_id(3)),markerArray(i),'MarkerSize',markerSize,'MarkerFaceColor',colorArray(i))
end

% 在XY平面上添加投影点
for i = 1:length(features)
    plot3(features(i,index_id(1)), features(i,index_id(2)), 0, ...
          projectionMarkerArray(speciesNum(i)), 'MarkerSize', markerSize, 'MarkerEdgeColor', colorArray(speciesNum(i)));
end

% 在YZ平面上添加投影点
for i = 1:length(features)
    plot3(0, features(i,index_id(2)), features(i,index_id(3)), ...
          projectionMarkerArray(speciesNum(i)), 'MarkerSize', markerSize, 'MarkerEdgeColor', colorArray(speciesNum(i)));
end

% 在XZ平面上添加投影点
for i = 1:length(features)
    plot3(features(i,index_id(1)), 0, features(i,index_id(3)), ...
          projectionMarkerArray(speciesNum(i)), 'MarkerSize', markerSize, 'MarkerEdgeColor', colorArray(speciesNum(i)));
end

hold off
grid on
view([60,24])
xlabel('Pansharpened In GF1 Model')
ylabel('Pansharpened In QB Model')
zlabel('Pansharpened In WV4 Model')
title('真实标记及投影')
set(gca,'FontSize',12)
set(gca,'FontWeight','Bold')

% 添加图例
legend('GF1 Image', 'QB Image', 'WV4 Image','GF1 Projection','QB Projection','WV4 Projection')


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