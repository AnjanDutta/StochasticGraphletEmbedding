function [] = check_hash_function(fh)

user = getenv('USER');

switch user
    case 'root'
        workhome = '/home/anjan/';
    case 'dag'
        workhome = '/home/anjan/';
end;

addpath(genpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/matlab_bgl']));
data_dir = '/home/anjan/Workspace/Graphlets';

nn = (2:11)';
ne = [nn-1,nn.*(nn-1)/2];

filenames = {'graph2c.txt','graph3c.txt','graph4c.txt','graph5c.txt',...
    'graph6c.txt','graph7c.txt','graph8c.txt','graph9c.txt','graph10c.txt',...
    'graph11c.txt'};

for ine = 1:10
    
    fprintf('Edges: %d. ', ine);
    
    tic;
    
    idx_file = find(ine >= ne(:,1) & ine <= ne(:,2))';
    
    edges = [];
    nvertices = [];
    
    for ifile = idx_file
        filename = fullfile(data_dir,filenames{ifile});
        if(~exist(filename,'file'))
            continue;
        end;
        
        [nvertices_,edges_] = read_graph(filename,ine);
        edges = [edges;edges_];
        nvertices = [nvertices;nvertices_];        
    end;
    
    ngraphs = size(edges,1);
    hcs = [];
    
    for ig = 1:ngraphs
        A = sparse(edges(ig,1:2:end),edges(ig,2:2:end),1,nvertices(ig),nvertices(ig));
        switch fh
            case 'degree_nodes'
                hc = sort(degree_nodes(double(A|A')))';
            case 'core_numbers'
                hc = sort(core_numbers(double(A|A')))';
            case 'clustering_coefficients'
                hc = sort(clustering_coefficients(double(A|A')))';
            case 'betweenness_centrality'
                hc = sort(betweenness_centrality(double(A|A')))';
        end;
        hcs = [hcs;[hc,zeros(1,2*ine-length(hc))]];
    end;
    
    [unique_bcs, ia, ic] = unique(hcs,'rows');
    
%     idx = setdiff(1:size(hcs,1), ia(sum(bsxfun(@eq,ic,(1:max(ic))))<=1));        
%     for ig = 1:length(idx)
%         id = idx(ig);
%         A = sparse(edges(id,1:2:end),edges(id,2:2:end),1,nvertices(id),nvertices(id));
%         h = draw_graphs_from_adjmat(A);
%         fpdf = ['graphlet_',num2str(ine),'_',fh,'_duplicate_',num2str(ic(idx(ig))),'_1.pdf'];
%         if(exist(fpdf,'file'))
%             fpdf = strrep(fpdf,'_1.pdf','_2.pdf');
%         end;        
%         saveas(h,fpdf);
%         disp([fpdf,' ',num2str(hcs(id,:))]);
%         delete(h);
%     end;
        
%     table = tabulate(ic);
%     dlmwrite(sprintf('Paths_%d.txt',ine),table);

    ng = size(hcs,1) ; % Number of Graphs
    nc = size(hcs,1)-size(unique_bcs,1) ; % Number of Collisions
    if(ng<2)  % Probability of Collisions
        pc = nc;
    else
        pc = nc / nchoosek(ng,2) ;
    end;    
    
    fprintf('Number of Graphs: %d. ', ng) ;
    fprintf('Number of Collisions: %d. ', nc) ;
    fprintf('Probability of Collisions: %d. ', pc) ;
    
    toc;
    
end;
