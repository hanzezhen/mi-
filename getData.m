function [Data,Label] = getData(currentPersonId)
% 读入数据
%prefix = (['TrainData\S100\']);
currentPersonId =1
prefix = (['TrainData\S' num2str(currentPersonId) '\']);
d = dir([prefix,'*.mat']);
time = 3;
offlength = 0.5;
dataLen = time*250;
sampleTime = time-offlength;
srate = 250;
totalFlt = [1,40];
%selectChannel = 1:1:60; 
selectChannel = [18 19 26 27 28 29 30]; 
%preprocessFilter = getPreFilter(srate);

for i=1:length(d)
    load([prefix,d(i).name]);
    [~, B] = find(data(65,:) ~= 0);
    label = data(65,B([2:2:80])); % 标签
    start = B([2:2:80]);
    for j = 1:40    
        rawdata = data(1:end-1, start(j) + offlength*srate:start(j) + time*srate - 1);
        %rawdata = preprocess(preprocessFilter,selectChannel,rawdata);      
        Data(:,:,j) = preProccess(srate,sampleTime,rawdata,totalFlt);
    end
    DATA(:,:,40*(i-1) + 1 : 40*i) = Data;
    LABEL(40*(i-1) + 1 : 40*i) = label;
end
end

function preprocessFilter = getPreFilter(srate)
Fo = 50;
Q = 35;
BW = (Fo/(srate/2))/Q;
[preprocessFilter.B,preprocessFilter.A] = iircomb(srate/Fo,BW,'notch');
end

function data = preprocess(preprocessFilter,selectChannel,data)

data = data(selectChannel,:);

data = filtfilt(preprocessFilter.B,preprocessFilter.A,data.');

data = data.';

end
