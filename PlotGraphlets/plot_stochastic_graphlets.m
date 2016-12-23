addpath('/home/anjan/Dropbox/Personal/Workspace/AdditionalTools/random_graphlet1') ;
addpath('/home/anjan/Dropbox/Personal/Workspace/MATLABTools/StochasticGraphletEmbedding/HashFunctionGraphlets') ;
n = 50 ;
A = ones(n) ;
A(1:n+1:n*n) = 0 ; % A is a fully connected graph
[I,J] = find(A) ;
edges = uint32([I,J]) ;
M = uint32(1) ;
T = uint32(40) ;
graphlets = generate_random_graphlets(edges,uint32(1),uint32(40)) ;
for i = 1:length(graphlets)
    adjmat_graphlet = sparse(double(graphlets{i}(1:2:end)), double(graphlets{i}(2:2:end)), 1, n, n) ;
    adjmat_graphlet = adjmat_graphlet | adjmat_graphlet' ;
    adjmat_graphlet(~any(adjmat_graphlet,2), :) = [] ;  %rows
    adjmat_graphlet(:, ~any(adjmat_graphlet,1)) = [] ;  %columns
    h = draw_graphs_from_adjmat(adjmat_graphlet) ;
    saveas(h, ['graphlet_', sprintf('%02d', i),'.pdf']) ;
    delete(h) ;
end ;