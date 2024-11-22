gridRows = 4;
gridCols = 4;
valueGrid = zeros(gridRows, gridCols);
convergenceThreshold = 0.001;
discountFactor = 0.9;
goalState = [4, 4];
trapState = [2, 3];

stateRewards = zeros(gridRows, gridCols);
stateRewards(goalState(1), goalState(2)) = 10;
stateRewards(trapState(1), trapState(2)) = -10;

actionOffsets = [0, -1; 0, 1; -1, 0; 1, 0];
numActions = size(actionOffsets, 1);

while true
    maxChange = 0;
    updatedValueGrid = valueGrid;
    
    for row = 1:gridRows
        for col = 1:gridCols
            if (row == goalState(1) && col == goalState(2)) || (row == trapState(1) && col == trapState(2))
                continue;
            end
            
            maxExpectedValue = -inf;
            for actionIdx = 1:numActions
                nextRow = row + actionOffsets(actionIdx, 1);
                nextCol = col + actionOffsets(actionIdx, 2);
                
                if nextRow < 1 || nextRow > gridRows || nextCol < 1 || nextCol > gridCols
                    nextRow = row;
                    nextCol = col;
                end
                
                actionValue = stateRewards(nextRow, nextCol) + discountFactor * valueGrid(nextRow, nextCol);
                maxExpectedValue = max(maxExpectedValue, actionValue);
            end
            
            updatedValueGrid(row, col) = maxExpectedValue;
            maxChange = max(maxChange, abs(valueGrid(row, col) - maxExpectedValue));
        end
    end
    
    if maxChange < convergenceThreshold
        break;
    end
    valueGrid = updatedValueGrid;
end

optimalPolicy = strings(gridRows, gridCols);
for row = 1:gridRows
    for col = 1:gridCols
        if (row == goalState(1) && col == goalState(2))
            optimalPolicy(row, col) = "Goal";
            continue;
        elseif (row == trapState(1) && col == trapState(2))
            optimalPolicy(row, col) = "Trap";
            continue;
        end
        
        maxExpectedValue = -inf;
        bestAction = "";
        for actionIdx = 1:numActions
            nextRow = row + actionOffsets(actionIdx, 1);
            nextCol = col + actionOffsets(actionIdx, 2);
            
            if nextRow < 1 || nextRow > gridRows || nextCol < 1 || nextCol > gridCols
                nextRow = row;
                nextCol = col;
            end
            
            actionValue = stateRewards(nextRow, nextCol) + discountFactor * valueGrid(nextRow, nextCol);
            if actionValue > maxExpectedValue
                maxExpectedValue = actionValue;
                switch actionIdx
                    case 1
                        bestAction = "Left";
                    case 2
                        bestAction = "Right";
                    case 3
                        bestAction = "Up";
                    case 4
                        bestAction = "Down";
                end
            end
        end
        optimalPolicy(row, col) = bestAction;
    end
end

disp("Optimal Value Function:");
disp(valueGrid);
disp("Optimal Policy:");
disp(optimalPolicy);
