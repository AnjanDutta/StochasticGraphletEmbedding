function [file_names,classes] = read_muta_cxl(filename)

% read the file as string
fs = fileread(filename);

expr = '<\w*\s\w*="([a-z0-9._]*)"\s\w*="(\w*)"/>\n';

tokens = regexp(fs,expr,'tokens');

if(isempty(tokens))    
    error('Error: Parsing nodes');
end;

file_names = cellfun(@(x) x{1},tokens,'UniformOutput',false)';
idx = strcmp(cellfun(@(x) x{2},tokens,'UniformOutput',false)','mutagen');

classes(idx) = 1;
classes(~idx) = 2;

clear idx tokens;

end