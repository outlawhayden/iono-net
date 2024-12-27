function coeffsToCSV

    seeds.count = 1000; % Modify as per original script intent

    stepRefinePow = 2;  
    ionoNharm = 6; 
    seeds.start = struct('ionosphere', 21, 'clutter', 61, 'PS', 41); 
    outputDir = '../noise_experiments/more_both';  % Directory to save CSV files
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);  % Create output directory if it doesn't exist
    end
    
    % Load the dataset once outside the loop
    matFname = 'radarSeries.mat';  
    S = load(matFname);  % Load the .mat file only once
    
    % Initialize matrices to accumulate nuStruct_withSpeckle, compl_ampls, and uscStruct_vals data
    nuStructMatrix = [];
    complAmplsMatrix = [];
    uscStruct_valsMatrix = [];

    % Loop over seeds and process them
    for iseed = 1:seeds.count
        [nuStructCol, complAmplsCol, uscStruct_valsCol] = processSeed(stepRefinePow, ionoNharm, seeds, iseed, S);
        
        % Accumulate columns for each seed
        nuStructMatrix = [nuStructMatrix, nuStructCol]; % Append new column
        complAmplsMatrix = [complAmplsMatrix, complAmplsCol]; % Append new column
        uscStruct_valsMatrix = [uscStruct_valsMatrix, uscStruct_valsCol]; % Append new column
    end
    
    % Export the matrices to CSV
    exportMatrixToCsv(nuStructMatrix, 'nuStruct_withSpeckle', outputDir);
    exportMatrixToCsv(complAmplsMatrix, 'compl_ampls', outputDir);
    exportMatrixToCsv(uscStruct_valsMatrix, 'uscStruct_vals', outputDir);  % Export the new matrix
    
    % Export dataset.meta.kPsi as a table to CSV
    kPsi = S.dataset.meta.kPsi;  % Extract kPsi
    exportMetaToCsv(kPsi, 'kPsi', outputDir);  % Export kPsi as CSV
    
    % Export dataset.meta.X, Z, and S as CSV files
    exportVectorToCsv(S.dataset.meta.X, 'meta_X', outputDir);
    exportVectorToCsv(S.dataset.meta.Z, 'meta_Z', outputDir);
    exportVectorToCsv(S.dataset.meta.S, 'meta_S', outputDir);

    % Export dataset.meta.setup to JSON
    metaSetup = S.dataset.meta.setup;
    filteredSetup = filterStructFields(metaSetup, {'steps', 'windowFun', 'createPsiImplFun', 'sumFun'});
    exportSetupToJson(filteredSetup, 'setup', outputDir);

    disp 'DONE'
end

function filteredStruct = filterStructFields(inputStruct, excludeFields)
    % Remove specified fields from a structure
    fields = fieldnames(inputStruct);
    for i = 1:numel(excludeFields)
        if isfield(inputStruct, excludeFields{i})
            inputStruct = rmfield(inputStruct, excludeFields{i});
        end
    end
    filteredStruct = inputStruct;
end

function exportSetupToJson(setupStruct, setupName, outputDir)
    % Convert the structure to a JSON file
    jsonStr = jsonencode(setupStruct);
    
    % Create filename
    jsonFname = sprintf('%s_%s.json', setupName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, jsonFname);
    
    % Write JSON to file
    fid = fopen(fullFname, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', fullFname);
    end
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    
    fprintf('Exported %s to %s\n', setupName, fullFname);
end

function exportVectorToCsv(vector, vectorName, outputDir)
    % Convert vector to table
    T = array2table(vector(:), 'VariableNames', {vectorName});
    
    % Create filename
    csvFname = sprintf('%s_%s.csv', vectorName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    % Write table to CSV
    writetable(T, fullFname);
    
    fprintf('Exported %s to %s\n', vectorName, fullFname);
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

function exportMetaToCsv(metaData, metaName, outputDir)
    % Convert meta data to table (assuming kPsi is a vector or matrix)
    T = array2table(metaData);
    
    % Create filename
    csvFname = sprintf('%s_%s.csv', metaName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    % Write table to CSV
    writetable(T, fullFname);
    
    fprintf('Exported %s to %s\n', metaName, fullFname);
end

function [nuStructCol, complAmplsCol, uscStruct_valsCol] = processSeed(stepRefinePow, ionoNharm, seeds, iseed, S)
    fprintf('Processing seed %d\n', iseed);   
    
    % Extract relevant data for this seed
    record = S.dataset.records{iseed};
    
    % Extract compl_ampls (complex amplitudes) and convert to a column vector
    compl_ampls = S.dataset.compl_ampls{iseed};
    complAmplsCol = compl_ampls(:);  % Ensure it's a column
    
    % Extract nuStruct_withSpeckle (complex values) and convert to a column vector
    nuStruct = record.nuStructs.withSpeckle.complVal;
    nuStructCol = nuStruct(:);  % Ensure it's a column
    
    % Extract uscStruct_vals (values) and convert to a column vector
    uscStruct_vals = record.uscStruct.vals;
    uscStruct_valsCol = uscStruct_vals(:);  % Ensure it's a column
end
