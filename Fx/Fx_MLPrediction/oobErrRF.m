% 定义一个目标函数用于贝叶斯优化算法进行优化。该函数应满足以下条件：
% 
% ① 以要调整的参数为输入；
% ② 使用TreeBagger训练随机森林。 在TreeBagger调用中，指定要调优的参数并指定返回袋外（out- out- bag）索引；
% ③ 根据中位数估计袋外分位数误差；
% ④ 返回袋外分位数误差

function oobErr = oobErrRF(params,X,ntree)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 300 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
% randomForest = TreeBagger(300,X,'MPG','Method','classification','OOBPrediction','on','MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS);
randomForest = TreeBagger(ntree,X,'train_data43','Method','classification','OOBPrediction','on','MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS);

oobErr = oobQuantileError(randomForest);
end
