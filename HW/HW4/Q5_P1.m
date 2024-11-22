gridRows = 4;
gridCols = 4;
valueMatrix = zeros(gridRows, gridCols);
actionPolicy = repmat("Left", gridRows, gridCols);

threshold = 0.01;
discountFactor = 0.9;
goalState = [4, 4];
trapState = [2, 3];

rewardMatrix = zeros(gridRows, gridCols);
rewardMatrix(goalState(1), goalState(2)) = 10;
rewardMatrix(trapState(1), trapState(2)) = -10;

possibleActions = ["Left", "Right", "Up", "Down"];
actionOffsets = [0, -1; 0, 1; -1, 0; 1, 0];

computeNextState = @(x, y, actionIdx) deal(...
    max(1, min(gridRows, x + actionOffsets(actionIdx, 1))), ...
    max(1, min(gridCols, y + actionOffsets(actionIdx, 2))));

while true
    while true
        maxDelta = 0;
        for row = 1:gridRows
            for col = 1:gridCols
                if isequal([row, col], goalState) || isequal([row, col], trapState)
                    continue;
                end
                oldValue = valueMatrix(row, col);
                actionIdx = find(possibleActions == actionPolicy(row, col));
                [nextRow, nextCol] = computeNextState(row, col, actionIdx);
                valueMatrix(row, col) = rewardMatrix(nextRow, nextCol) + ...
                                        discountFactor * valueMatrix(nextRow, nextCol);
                maxDelta = max(maxDelta, abs(oldValue - valueMatrix(row, col)));
            end
        end
        if maxDelta < threshold
            break;
        end
    end

    isPolicyStable = true;
    for row = 1:gridRows
        for col = 1:gridCols
            if isequal([row, col], goalState)
                actionPolicy(row, col) = "Goal";
                continue;
            elseif isequal([row, col], trapState)
                actionPolicy(row, col) = "Trap";
                continue;
            end
            
            previousAction = actionPolicy(row, col);
            actionValues = zeros(1, numel(possibleActions));
            for actionIdx = 1:numel(possibleActions)
                [nextRow, nextCol] = computeNextState(row, col, actionIdx);
                actionValues(actionIdx) = rewardMatrix(nextRow, nextCol) + ...
                                          discountFactor * valueMatrix(nextRow, nextCol);
            end
            [maxValue, optimalActionIdx] = max(actionValues);
            actionPolicy(row, col) = possibleActions(optimalActionIdx);
            
            if previousAction ~= actionPolicy(row, col)
                isPolicyStable = false;
            end
        end
    end
    
    if isPolicyStable
        break;
    end
end

disp("Optimal Value Matrix:");
disp(valueMatrix);
disp("Optimal Action Policy:");
disp(actionPolicy);
