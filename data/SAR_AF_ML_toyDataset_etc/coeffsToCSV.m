function coeffsToCSV
    % ================================================================
    % coeffsToCSV
    % ------------------------------------------------
    % This function processes a radar dataset contained in 'radarSeries.mat'
    % and splits it into training and testing sets. For each seed/sample:
    %   - Extracts three structures: nuStruct, complAmpls, and uscStruct_vals
    %   - Appends these to training or testing matrices
    %   - Exports the results to CSV files
    % It also saves metadata (kPsi, X, Z, S, setup) into CSV/JSON.
    %
    % ================================================================

    % ----------------------------
    % PARAMETERS AND SETTINGS
    % ----------------------------

    % Total number of seeds to process (samples in dataset).
    % Adjust this number to match dataset size or experiment scale.
    seeds.count = 10000; 
    
    % Ratio of training data (85% training, 15% testing here).
    trainRatio = 0.85;   
    trainCount = round(seeds.count * trainRatio); % Number of training samples
    testCount = seeds.count - trainCount;         % Remaining go to testing

    % Refinement power (dataset specific parameter).
    stepRefinePow = 2;  
    
    % Number of harmonic terms in the ionosphere representation.
    ionoNharm = 6; 

    % Starting indices for each seed type (ionosphere, clutter, PS).
    % These values influence reproducibility/randomization.
    seeds.start = struct('ionosphere', 21, 'clutter', 61, 'PS', 41); 

    % Output directory for generated CSV and JSON files.
    outputDir = '/home/houtlaw/iono-net/data/aug25';  
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);  % Create directory if missing
    end
    
    % ----------------------------
    % LOAD DATASET
    % ----------------------------

    % The radar dataset is expected to be stored in a MAT file.
    % Load once (outside loop) for efficiency.
    matFname = 'radarSeries.mat';  
    S = load(matFname);  

    % ----------------------------
    % INITIALIZE STORAGE MATRICES
    % ----------------------------
    % Training matrices will store column-wise concatenation of extracted data
    trainNuStructMatrix = [];
    trainComplAmplsMatrix = [];
    trainUscStruct_valsMatrix = [];
    
    % Testing matrices
    testNuStructMatrix = [];
    testComplAmplsMatrix = [];
    testUscStruct_valsMatrix = [];

    % ----------------------------
    % MAIN PROCESSING LOOP
    % ----------------------------
    % Iterate through all seeds (samples) and process one by one.
    % Each seed produces three column vectors:
    %   - nuStructCol
    %   - complAmplsCol
    %   - uscStruct_valsCol
    %
    % Depending on whether the index is within training or testing set,
    % append to the corresponding matrix.
    for iseed = 1:seeds.count
        % Extract columns for current seed
        [nuStructCol, complAmplsCol, uscStruct_valsCol] = processSeed(stepRefinePow, ionoNharm, seeds, iseed, S);

        if iseed <= trainCount
            % Append to training matrices
            trainNuStructMatrix = [trainNuStructMatrix, nuStructCol]; 
            trainComplAmplsMatrix = [trainComplAmplsMatrix, complAmplsCol]; 
            trainUscStruct_valsMatrix = [trainUscStruct_valsMatrix, uscStruct_valsCol];
        else
            % Append to testing matrices
            testNuStructMatrix = [testNuStructMatrix, nuStructCol]; 
            testComplAmplsMatrix = [testComplAmplsMatrix, complAmplsCol]; 
            testUscStruct_valsMatrix = [testUscStruct_valsMatrix, uscStruct_valsCol];
        end
    end
    
    % ----------------------------
    % EXPORT TRAINING MATRICES
    % ----------------------------
    exportMatrixToCsv(trainNuStructMatrix, 'train_nuStruct_withSpeckle', outputDir);
    exportMatrixToCsv(trainComplAmplsMatrix, 'train_compl_ampls', outputDir);
    exportMatrixToCsv(trainUscStruct_valsMatrix, 'train_uscStruct_vals', outputDir);

    % ----------------------------
    % EXPORT TEST MATRICES
    % ----------------------------
    exportMatrixToCsv(testNuStructMatrix, 'test_nuStruct_withSpeckle', outputDir);
    exportMatrixToCsv(testComplAmplsMatrix, 'test_compl_ampls', outputDir);
    exportMatrixToCsv(testUscStruct_valsMatrix, 'test_uscStruct_vals', outputDir);

    % ----------------------------
    % EXPORT METADATA
    % ----------------------------

    % kPsi (Fourier Psi coefficients) from dataset metadata
    kPsi = S.dataset.meta.kPsi;  
    exportMetaToCsv(kPsi, 'kPsi', outputDir);  
    
    % Spatial metadata vectors
    exportVectorToCsv(S.dataset.meta.X, 'meta_X', outputDir);
    exportVectorToCsv(S.dataset.meta.Z, 'meta_Z', outputDir);
    exportVectorToCsv(S.dataset.meta.S, 'meta_S', outputDir);

    % Export setup struct (excluding large/unnecessary fields) as JSON
    metaSetup = S.dataset.meta.setup;
    filteredSetup = filterStructFields(metaSetup, {'steps', 'windowFun', 'createPsiImplFun', 'sumFun'});
    exportSetupToJson(filteredSetup, 'setup', outputDir);

    disp 'DONE'
end

% ================================================================
% HELPER FUNCTIONS
% ================================================================

function filteredStruct = filterStructFields(inputStruct, excludeFields)
    % Remove specified fields from a structure (to avoid exporting functions
    % or large unnecessary fields into JSON).
    fields = fieldnames(inputStruct);
    for i = 1:numel(excludeFields)
        if isfield(inputStruct, excludeFields{i})
            inputStruct = rmfield(inputStruct, excludeFields{i});
        end
    end
    filteredStruct = inputStruct;
end

function exportSetupToJson(setupStruct, setupName, outputDir)
    % Convert MATLAB structure into JSON format and write to file.
    
    jsonStr = jsonencode(setupStruct);
    
    % File is timestamped to avoid overwriting past runs
    jsonFname = sprintf('%s_%s.json', setupName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, jsonFname);
    
    % Write JSON string to disk
    fid = fopen(fullFname, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', fullFname);
    end
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    
    fprintf('Exported %s to %s\n', setupName, fullFname);
end

function exportVectorToCsv(vector, vectorName, outputDir)
    % Save a 1D vector into a single-column CSV file.
    
    % Wrap vector into a table with column named vectorName
    T = array2table(vector(:), 'VariableNames', {vectorName});
    
    % Timestamped filename
    csvFname = sprintf('%s_%s.csv', vectorName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    writetable(T, fullFname);
    
    fprintf('Exported %s to %s\n', vectorName, fullFname);
end

function exportMatrixToCsv(matrix, matrixName, outputDir)
    % Save a 2D matrix into a CSV file.
    % Each column represents one sample/seed.
    
    T = array2table(matrix);
    
    csvFname = sprintf('%s_%s.csv', matrixName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    writetable(T, fullFname);
    
    fprintf('Exported %s matrix to %s\n', matrixName, fullFname);
end

function exportMetaToCsv(metaData, metaName, outputDir)
    % Save metadata (e.g., kPsi array) into a CSV file.
    % Assumes metadata is a vector or matrix.
    
    T = array2table(metaData);
    
    csvFname = sprintf('%s_%s.csv', metaName, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    writetable(T, fullFname);
    
    fprintf('Exported %s to %s\n', metaName, fullFname);
end

function [nuStructCol, complAmplsCol, uscStruct_valsCol] = processSeed(stepRefinePow, ionoNharm, seeds, iseed, S)
    % Extracts data for a given seed index from dataset S.
    % Returns column vectors for:
    %   - nuStructCol (speckle-augmented structural data)
    %   - complAmplsCol (complex amplitudes)
    %   - uscStruct_valsCol (uncorrected structure values)

    fprintf('Processing seed %d\n', iseed);   
    
    % Each seed corresponds to a record inside dataset.records cell array
    record = S.dataset.records{iseed};
    
    % Extract complex amplitudes (compl_ampls)
    compl_ampls = S.dataset.compl_ampls{iseed};
    complAmplsCol = compl_ampls(:);  % Force column vector
    
    % Extract nuStruct_withSpeckle (complex structure values)
    nuStruct = record.nuStructs.withSpeckle.complVal;
    nuStructCol = nuStruct(:);  
    
    % Extract uscStruct values (uncorrected structure values)
    uscStruct_vals = record.uscStruct.vals;
    uscStruct_valsCol = uscStruct_vals(:);  
end
