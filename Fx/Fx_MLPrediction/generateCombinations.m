function comboSave = generateCombinations(arr, k, combo, indexNum)
    comboSave = cell(0, 1);

    if k == 0
        disp(combo);
        comboSave{end + 1} = combo;
    elseif indexNum == 0
        return;
    else
        combo(k) = arr(indexNum);
        % 累积组合
        subCombinations1 = generateCombinations(arr, k - 1, combo, indexNum - 1);
        subCombinations2 = generateCombinations(arr, k, combo, indexNum - 1);
        comboSave = [comboSave; subCombinations1; subCombinations2];
    end
end
