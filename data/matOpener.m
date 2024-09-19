folder = '/Users/haydenoutlaw/Documents/Research/SAR-Network/iono-net/data/SAR_AF_ML_toyDataset_etc';  % You specify this!
fullMatFileName = fullfile(folder,  'radarSeries.mat');
if ~exist(fullMatFileName, 'file')
  message = sprintf('%s does not exist', fullMatFileName);
  uiwait(warndlg(message));
else
  s = load(fullMatFileName);
end

%whos('-file', fullMatFileName)

whos dataset
%fieldnames(dataset)
%disp(dataset)
openvar('dataset')