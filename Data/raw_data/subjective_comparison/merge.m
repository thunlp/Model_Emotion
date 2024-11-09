subjects = readtable('prolific_info.csv');
n = [];
for i = 1:height(subjects)
    if strcmp(subjects.Status{i},'APPROVED')
        n = [n i];
    end
end
subjects = subjects(n,:);
p = dir('/Users/liming/Desktop/学术/项目/Emotion_NLP/Data/raw_data/subjective_comparison');
rating = NaN(378,height(subjects));
for i = 1:height(subjects)
    for n = 1:length(p)
        if ~isempty(strfind(p(n).name, subjects.ParticipantId{i}))
            filename = [p(n).folder filesep p(n).name];
            temp = ImportData(filename);
            subjects.data{i} = temp;
            rating(:,i) = 10-temp;
        end
    end
end
clear p n i opts filename temp;

%%
filename = '/Users/liming/Desktop/学术/项目/Emotion_NLP/Data/raw_data/subjective_comparison/emotions.txt';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, '%s%[^\n\r]', 'Delimiter', {''}, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
emotions = dataArray{1,1};
EmoPair = nchoosek(emotions,2);
data = table(EmoPair);
data.DistMean = mean(rating,2,'omitnan');
data.DistStd = std(rating,0,2,'omitnan');
data.raw = rating;
clearvars filename fileID dataArray rating emotions EmoPair ans;

%%
writetable(data,'/Users/liming/Desktop/学术/项目/Emotion_NLP/Data/raw_data/subjective_comparison/data.xlsx')
writetable(subjects,'/Users/liming/Desktop/学术/项目/Emotion_NLP/Data/raw_data/subjective_comparison/subjects.xlsx')