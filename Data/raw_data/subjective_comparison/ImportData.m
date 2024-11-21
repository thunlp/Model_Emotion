function data = ImportData(filename, startRow, endRow)
%IMPORTFILE ���ı��ļ��е���ֵ������Ϊ�����롣
%   A = IMPORTFILE(FILENAME) ��ȡ�ı��ļ� FILENAME ��Ĭ��ѡ����Χ�����ݡ�
%
%   A = IMPORTFILE(FILENAME, STARTROW, ENDROW) ��ȡ�ı��ļ� FILENAME �� STARTROW
%   �е� ENDROW ���е����ݡ�
%
% Example:
%   a = importfile('5b987c55bb32a600018302dc_PRaM_2021-12-14_20h35.31.231.csv', 2, 379);
%
%    ������� TEXTSCAN��

% �� MATLAB �Զ������� 2021/12/16 12:50:31

%% ��ʼ��������
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% ����������Ϊ�ı���ȡ:
% �й���ϸ��Ϣ������� TEXTSCAN �ĵ���
formatSpec = '%s%[^\n\r]';

%% ���ı��ļ���
fileID = fopen(filename,'r','n','UTF-8');
% ���� BOM (�ֽ�˳����)��
fseek(fileID, 3, 'bof');

%% ���ݸ�ʽ��ȡ�����С�
% �õ��û������ɴ˴������õ��ļ��Ľṹ����������ļ����ִ����볢��ͨ�����빤���������ɴ��롣
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% �ر��ı��ļ���
fclose(fileID);

%% ��������ֵ�ı���������ת��Ϊ��ֵ��
% ������ֵ�ı��滻Ϊ NaN��
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

% ������Ԫ�������е��ı�ת��Ϊ��ֵ���ѽ�����ֵ�ı��滻Ϊ NaN��
rawData = dataArray{1};
for row=1:size(rawData, 1)
    % ����������ʽ�Լ�Ⲣɾ������ֵǰ׺�ͺ�׺��
    regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
    try
        result = regexp(rawData(row), regexstr, 'names');
        numbers = result.numbers;
        
        % �ڷ�ǧλλ���м�⵽���š�
        invalidThousandsSeparator = false;
        if numbers.contains(',')
            thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
            if isempty(regexp(numbers, thousandsRegExp, 'once'))
                numbers = NaN;
                invalidThousandsSeparator = true;
            end
        end
        % ����ֵ�ı�ת��Ϊ��ֵ��
        if ~invalidThousandsSeparator
            numbers = textscan(char(strrep(numbers, ',', '')), '%f');
            numericData(row, 1) = numbers{1};
            raw{row, 1} = numbers{1};
        end
    catch
        raw{row, 1} = rawData{row};
    end
end


%% ������ֵԪ���滻Ϊ NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % ���ҷ���ֵԪ��
raw(R) = {NaN}; % �滻����ֵԪ��

%% �����������
a = cell2mat(raw);
b = ImportIndex(filename);
[~,I] = sort(b);
data = a(I);

function index = ImportIndex(filename, startRow, endRow)
%IMPORTFILE1 ���ı��ļ��е���ֵ������Ϊ�����롣
%   B987C55BB32A600018302DCPRAM2021121420H35 = IMPORTFILE1(FILENAME) ��ȡ�ı��ļ�
%   FILENAME ��Ĭ��ѡ����Χ�����ݡ�
%
%   B987C55BB32A600018302DCPRAM2021121420H35 = IMPORTFILE1(FILENAME,
%   STARTROW, ENDROW) ��ȡ�ı��ļ� FILENAME �� STARTROW �е� ENDROW ���е����ݡ�
%
% Example:
%   b987c55bb32a600018302dcPRaM2021121420h35 = importfile1('5b987c55bb32a600018302dc_PRaM_2021-12-14_20h35.31.231.csv', 2, 379);
%
%    ������� TEXTSCAN��

% �� MATLAB �Զ������� 2021/12/16 12:54:10

%% ��ʼ��������
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% ÿ���ı��еĸ�ʽ:
%   ��8: ˫����ֵ (%f)
% �й���ϸ��Ϣ������� TEXTSCAN �ĵ���
formatSpec = '%*s%*s%*s%*s%*s%*s%*s%f%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]';

%% ���ı��ļ���
fileID = fopen(filename,'r','n','UTF-8');
% ���� BOM (�ֽ�˳����)��
fseek(fileID, 3, 'bof');

%% ���ݸ�ʽ��ȡ�����С�
% �õ��û������ɴ˴������õ��ļ��Ľṹ����������ļ����ִ����볢��ͨ�����빤���������ɴ��롣
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% �ر��ı��ļ���
fclose(fileID);

%% ���޷���������ݽ��еĺ���
% �ڵ��������δӦ���޷���������ݵĹ�����˲�����������롣Ҫ�����������޷���������ݵĴ��룬�����ļ���ѡ���޷������Ԫ����Ȼ���������ɽű���

%% �����������
index = [dataArray{1:end-1}];

