function error = classificationErrorRF(params, X, Y)
    % 创建随机森林分类器
    numTrees = 100;
    minLeafSize = params.minLS;
    numPredictorsToSample = params.numPTS;
    Mdl = TreeBagger(numTrees, X, Y, ...
                     'Method', 'classification', ...
                     'MinLeafSize', minLeafSize, ...
                     'NumPredictorsToSample', numPredictorsToSample);

    % 使用随机森林进行预测
    Y_pred = predict(Mdl, X);

    % 计算分类错误率
    error = sum(~strcmp(Y_pred, Y)) / numel(Y);
end
