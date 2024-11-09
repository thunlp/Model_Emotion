function data = ImportData(filename, startRow, endRow)
%IMPORTFILE 将文本文件中的数值数据作为矩阵导入。
%   A = IMPORTFILE(FILENAME) 读取文本文件 FILENAME 中默认选定范围的数据。
%
%   A = IMPORTFILE(FILENAME, STARTROW, ENDROW) 读取文本文件 FILENAME 的 STARTROW
%   行到 ENDROW 行中的数据。
%
% Example:
%   a = importfile('5b987c55bb32a600018302dc_PRaM_2021-12-14_20h35.31.231.csv', 2, 379);
%
%    另请参阅 TEXTSCAN。

% 由 MATLAB 自动生成于 2021/12/16 12:50:31

%% 初始化变量。
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% 将数据列作为文本读取:
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%s%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r','n','UTF-8');
% 跳过 BOM (字节顺序标记)。
fseek(fileID, 3, 'bof');

%% 根据格式读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% 关闭文本文件。
fclose(fileID);

%% 将包含数值文本的列内容转换为数值。
% 将非数值文本替换为 NaN。
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

% 将输入元胞数组中的文本转换为数值。已将非数值文本替换为 NaN。
rawData = dataArray{1};
for row=1:size(rawData, 1)
    % 创建正则表达式以检测并删除非数值前缀和后缀。
    regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
    try
        result = regexp(rawData(row), regexstr, 'names');
        numbers = result.numbers;
        
        % 在非千位位置中检测到逗号。
        invalidThousandsSeparator = false;
        if numbers.contains(',')
            thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
            if isempty(regexp(numbers, thousandsRegExp, 'once'))
                numbers = NaN;
                invalidThousandsSeparator = true;
            end
        end
        % 将数值文本转换为数值。
        if ~invalidThousandsSeparator
            numbers = textscan(char(strrep(numbers, ',', '')), '%f');
            numericData(row, 1) = numbers{1};
            raw{row, 1} = numbers{1};
        end
    catch
        raw{row, 1} = rawData{row};
    end
end


%% 将非数值元胞替换为 NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % 查找非数值元胞
raw(R) = {NaN}; % 替换非数值元胞

%% 创建输出变量
a = cell2mat(raw);
b = ImportIndex(filename);
[~,I] = sort(b);
data = a(I);

function index = ImportIndex(filename, startRow, endRow)
%IMPORTFILE1 将文本文件中的数值数据作为矩阵导入。
%   B987C55BB32A600018302DCPRAM2021121420H35 = IMPORTFILE1(FILENAME) 读取文本文件
%   FILENAME 中默认选定范围的数据。
%
%   B987C55BB32A600018302DCPRAM2021121420H35 = IMPORTFILE1(FILENAME,
%   STARTROW, ENDROW) 读取文本文件 FILENAME 的 STARTROW 行到 ENDROW 行中的数据。
%
% Example:
%   b987c55bb32a600018302dcPRaM2021121420h35 = importfile1('5b987c55bb32a600018302dc_PRaM_2021-12-14_20h35.31.231.csv', 2, 379);
%
%    另请参阅 TEXTSCAN。

% 由 MATLAB 自动生成于 2021/12/16 12:54:10

%% 初始化变量。
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% 每个文本行的格式:
%   列8: 双精度值 (%f)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%*s%*s%*s%*s%*s%*s%*s%f%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r','n','UTF-8');
% 跳过 BOM (字节顺序标记)。
fseek(fileID, 3, 'bof');

%% 根据格式读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% 关闭文本文件。
fclose(fileID);

%% 对无法导入的数据进行的后处理。
% 在导入过程中未应用无法导入的数据的规则，因此不包括后处理代码。要生成适用于无法导入的数据的代码，请在文件中选择无法导入的元胞，然后重新生成脚本。

%% 创建输出变量
index = [dataArray{1:end-1}];

