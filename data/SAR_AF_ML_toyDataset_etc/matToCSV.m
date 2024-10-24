function matToCSV

    seeds.count = 1000; % Modify as per original script intent

    stepRefinePow = 2;  
    ionoNharm = 6; 
    seeds.start = struct('ionosphere', 21, 'clutter', 61, 'PS', 41); 
    outputDir = 'radar_series_csv';  % Directory to save CSV files
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);  % Create output directory if it doesn't exist
    end
    
    % Load the dataset once outside the loop
    matFname = 'radarSeries.mat';  
    S = load(matFname);  % Load the .mat file only once
    
    % Loop over seeds and process them
    for iseed = 1:seeds.count
        createDataForRangeOfSeeds(stepRefinePow, ionoNharm, seeds, iseed, outputDir, S); 
    end
    
    disp 'DONE'
end

function createDataForRangeOfSeeds(stepRefinePow, ionoNharm, seeds, iseed, outputDir, S)
    fprintf('Processing seed %d\n', iseed);   
    
    % Extract relevant data for this seed
    setup = S.dataset.meta.setup;
    record = S.dataset.records{iseed};
    meta = S.dataset.meta;
    
    % Extract compl_ampls (complex amplitudes)
    compl_ampls = S.dataset.compl_ampls{iseed};
    
    % Export each relevant struct to separate CSV files
    exportStructToCsv(record.nuStructs.withSpeckle, meta.Z, 'nuStruct_withSpeckle', iseed, outputDir);
    exportStructToCsv(record.nuStructs.withoutSpeckle, meta.Z, 'nuStruct_withoutSpeckle', iseed, outputDir);
    exportStructToCsv(record.storedPsi, meta.S, 'storedPsi', iseed, outputDir);
    exportStructToCsv(record.storedPsi_dd_Val, meta.S(2:end-1), 'storedPsi_dd_Val', iseed, outputDir);
    exportStructToCsv(record.uscStruct.vals, meta.X, 'uscStruct_vals', iseed, outputDir);
    
    % Export compl_ampls (complex amplitudes) to a CSV file
    exportStructToCsv(compl_ampls, [], 'compl_ampls', iseed, outputDir);
    
    disp(['Data for seed ' num2str(iseed) ' exported to CSV.']);
end

function exportStructToCsv(data, coord, structName, iseed, outputDir)
    % Check if data is a struct
    if isstruct(data)
        % Handle specific structs like storedPsi
        if isfield(data, 'arg') && isfield(data, 'val')
            % For storedPsi, export both 'arg' and 'val'
            T = table(data.arg(:), data.val(:), 'VariableNames', {'Arg', 'Val'});
        elseif isfield(data, 'complVal')
            % For nuStructs (with 'complVal' and coordinate 'coord')
            T = table(coord(:), abs(data.complVal(:)), real(data.complVal(:)), imag(data.complVal(:)), ...
                      'VariableNames', {'Coord', 'AbsComplVal', 'RealComplVal', 'ImagComplVal'});
        else
            error('Unsupported struct type');
        end
    elseif isnumeric(data) && isempty(coord)
        % Handle abs_ampls (complex amplitudes) when coord is empty
        T = table(real(data(:)), imag(data(:)), abs(data(:)), ...
                  'VariableNames', {'RealPart', 'ImagPart', 'AbsValue'});
    else
        % Specific handling for storedPsi_dd_Val
        if strcmp(structName, 'storedPsi_dd_Val')
            % Remove first and last elements from data to match coord size
            data = data(2:end-1);
        end
        
        % Check if coord and data sizes are compatible
        len_coord = length(coord(:));
        len_data = length(data(:));
        fprintf('Length of coord: %d\n', len_coord);
        fprintf('Length of data: %d\n', len_data);
        
        if len_coord ~= len_data
            error('Coord and data lengths do not match. Please check your input sizes.');
        end
        
        % For simple arrays like storedPsi_dd_Val, uscStruct.vals
        T = table(coord(:), data(:), 'VariableNames', {'Coord', structName});
    end
    
    % Create filename
    csvFname = sprintf('%s_seed%d_%s.csv', structName, iseed, datestr(now, 'yyyymmdd_HHMMSS'));
    fullFname = fullfile(outputDir, csvFname);
    
    % Write table to CSV
    writetable(T, fullFname);
    
    fprintf('Exported %s to %s\n', structName, fullFname);
end
