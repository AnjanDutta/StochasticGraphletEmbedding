function [nvertices,edges] = read_graph(filename,nedges)

nvertices = [];
edges = [];

% read the file
fp = fopen(filename,'r');

while(~feof(fp))

    tokens = textscan(fp,'Graph %*d, order %*d\n%d %d\n%[^Graph]',100000);
    idx = tokens{2} == nedges;
    if(~nnz(idx))
        continue;
    end;
    nvertices = [nvertices;double(tokens{1}(idx))];
    edges_ = cell2mat(cellfun(@str2num,tokens{3}(idx,:),'UniformOutput',false))+1;
    edges = [edges;edges_];

end;

fclose(fp);

end