function coeffsToCSV

    seeds.count = 10000; % Modify as per original script intent

    stepRefinePow = 2;  
    ionoNharm = 6; 
    seeds.start = struct('ionosphere', 21, 'clutter', 61, 'PS', 41); 
    outputDir = 'radar_coeffs_csv';  % Directory to save CSV files
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);  % Create output directory if it doesn't exist
    end
    
    % Load the dataset once outside the loop
    matFname = 'radarSeries.mat';  
    S = load(matFname);  % Load the .mat file only once
    
    % Initialize matrices to accumulate nuStruct_withSpeckle and compl_ampls data
    nuStructMatrix = [];
    complAmplsMatrix = [];

    % Loop over seeds and process them
    for iseed = 1:seeds.count
        [nuStructCol, complAmplsCol] = processSeed(stepRefinePow, ionoNharm, seeds, iseed, S);
        
        % Accumulate columns for each seed
        nuStructMatrix = [nuStructMatrix, nuStructCol]; % Append new column
        complAmplsMatrix = [complAmplsMatrix, complAmplsCol]; % Append new column
    end
    
    % Export the matrices to CSV
    exportMatrixToCsv(nuStructMatrix, 'nuStruct_withSpeckle', outputDir);
    exportMatrixToCsv(complAmplsMatrix, 'compl_ampls', outputDir);
    
    disp 'DONE'
end

function [nuStructCol, complAmplsCol] = processSeed(stepRefinePow, ionoNharm, seeds, iseed, S)
    fprintf('Processing seed %d\n', iseed);   
    
    % Extract relevant data for this seed
    record = S.dataset.records{iseed};
    
    % Extract compl_ampls (complex amplitudes) and convert to a column vector
    compl_ampls = S.dataset.compl_ampls{iseed};
    complAmplsCol = compl_ampls(:);  % Ensure it's a column
    
    % Extract nuStruct_withSpeckle (complex values) and convert to a column vector
    nuStruct = record.nuStructs.withSpeckle.complVal;
    nuStructCol = nuStruct(:);  % Ensure it's a column
end

function exportMatrixToCsv(matrix, matrixName, outputDir)
    % Create table for the matrix
    T = array2table(matrix);
    
    % Create filename
    csvFname = sprintf('%s_%s.csv', matrixName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    % Write table to CSV
    writetable(T, fullFname);
    
    fprintf('Exported %s matrix to %s\n', matrixName, fullFname);
end
