function [vertices,nvertices,atrvertices,edges,nedges,atredges] = read_grec_gxl(filename)

% read the file as string
fs = fileread(filename);

% split them with delimeter '_' to get the different parameters
% get the node ids and labels

node_expr = ['<node\sid="(\d*)">\n',...
    '<\w*\s\w*="x"><\w*>([0-9.-]*)</\w*></\w*>\n',...
    '<\w*\s\w*="y"><\w*>([0-9.-]*)</\w*></\w*>\n',...
    '<\w*\s\w*="type"><\w*>(\w*)</\w*></\w*>\n',...
    '</node>'];
edge_expr = ['<edge\s\w*="([0-9]*)"\s\w*="([0-9]*)">\n',...
    '<\w*\s\w*="\w*"><\w*>([0-9]*)</\w*></\w*>\n',...
    '<\w*\s\w*="\w*"><\w*>(\w*)</\w*></\w*>\n',...
    '<\w*\s\w*="\w*"><\w*>([0-9.-]*)</\w*></\w*>\n',...
    '(?:<\w*\s\w*="\w*"><\w*>)?(\w*)(?:</\w*></\w*>\n)?',...
    '(?:<\w*\s\w*="\w*"><\w*>)?([0-9.-]*)(?:</\w*></\w*>\n)?',...
    '</edge>'];

tokens = regexp(fs,node_expr,'tokens');
if(isempty(tokens))    
    error('Error: Parsing nodes');
end;
nvertices = size(tokens,2);
vertices = zeros(nvertices,2);
atrvertices = cell(nvertices,1);
for i = 1:nvertices
    vertices(i,:) = [str2double(tokens{i}(2)),str2double(tokens{i}(3))];
    atrvertices{i} = tokens{i}{4};
end;

% get the source and target of edges and labels
tokens = regexp(fs,edge_expr,'tokens');

if(isempty(tokens))
    error('Error: Parsing edges');
end;
nedges = size(tokens,2);
edges = zeros(nedges,2);
atredges = cell(nedges,1);

for i = 1:nedges
    edges(i,:) = [str2double(tokens{i}(1)),str2double(tokens{i}(2))] + 1;    
    if(strcmp(tokens{i}(6),'line'))
        atredges{i} = [str2double(tokens{i}(3)),tokens{i}(6),str2double(tokens{i}(7)),tokens{i}(4),str2double(tokens{i}(5))];
    else
        atredges{i} = [str2double(tokens{i}(3)),tokens{i}(4),str2double(tokens{i}(5)),tokens{i}(6),str2double(tokens{i}(7))];
    end;
end;

end