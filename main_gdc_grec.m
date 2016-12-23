function main_gdc_grec(MAX1, MAX2)

path_dependency;

user = getenv('USER');

switch user
    case 'root'
        workhome = '/home/adutta/Workspace/';
    case 'dag'
        workhome = '/home/dag/Workspace/';
    case 'adutta'
        workhome = '/home/dag/adutta/';
end;

GDCroot = [workhome,'GDC/'];

file_node_vocab = fullfile(GDCroot,'Data/GREC','node_vocab.mat');
file_edge_vocab = fullfile(GDCroot,'Data/GREC','edge_vocab.mat');
file_hist_indices = fullfile(GDCroot,'Data/GREC','hist_indices.mat');
file_kernels = fullfile(GDCroot,'Data/GREC','kernels.mat');
file_results = fullfile(GDCroot,'Results/GREC','results.txt');
file_diss_matrix = fullfile(GDCroot,'Results/GREC',sprintf('diss_%d_%d.dat',MAX1,MAX2));
file_time = fullfile(GDCroot,'Results/GREC',sprintf('time_%d_%d.txt',MAX1,MAX2));

force_compute.vocab = true;
force_compute.hist_indices = true;
force_compute.kernels = true;

MAX1 = uint32(MAX1);
MAX2 = uint32(MAX2);

cc = [1 10 50 100 1000];
%% Prepare training and test set

% seeding randomization
rng(0);

dir_graphs = [GDCroot,'Database/GREC/data/'];

[train_graph_names,train_classes] = read_grec_cxl([dir_graphs,'train.cxl']);
[valid_graph_names,valid_classes] = read_grec_cxl([dir_graphs,'valid.cxl']);
[test_graph_names,test_classes] = read_grec_cxl([dir_graphs,'grec-test.cxl']);

graph_names = [train_graph_names;valid_graph_names;test_graph_names];

train_set = 1:length(train_graph_names);
valid_set = (1:length(valid_graph_names)) + length(train_set);
test_set = (1:length(test_graph_names)) + length(train_graph_names) + length(valid_graph_names);

ntrain_set = size(train_graph_names,1);
nvalid_set = size(valid_graph_names,1);
ntest_set = size(test_graph_names,1);
ngraphs = size(graph_names,1);

uniq_classes = unique(train_classes)';
w_str = [];
for ic = uniq_classes
    nc = nnz(train_classes == ic);
    w_str = [w_str,sprintf('-w%d %.2f ',ic,ntrain_set/nc)];
end;
%% First create a node and edge vocabulary

if(~exist(file_node_vocab,'file')||~exist(file_edge_vocab,'file')||...
        force_compute.vocab)
    
    cntrs_atrvertices = [];
    cntrs_atredges = [];
    
    fprintf('Creating node vocabulary...');
    
    for i = 1:ngraphs
        
        [~,~,atrvertices,~,~,atredges] =...
            read_grec_gxl(fullfile(dir_graphs,graph_names{i}));
        
        atredges = cellfun(@(x) [x{2},x{4}], atredges,'UniformOutput',false);
        
        cntrs_atrvertices = unique([cntrs_atrvertices;atrvertices]);
        cntrs_atredges = unique([cntrs_atredges;atredges]);
        
        clear atrvertices atredges;
        
    end;
    
    save(file_node_vocab,'cntrs_atrvertices');    
    save(file_edge_vocab,'cntrs_atredges');
   
    fprintf('Done.\n');
    
else
    
    fprintf('Node vocabulary already exist.\n');    
    
    load(file_node_vocab,'cntrs_atrvertices');
    load(file_edge_vocab,'cntrs_atredges');
    
end;
%% Now create histogram indices

temp = tic;

if(~exist(file_hist_indices,'file')||force_compute.hist_indices)
    
    hash_codes_uniq = cell(MAX2,1);
    idx_image = cell(MAX2,1);
    idx_bin = cell(MAX2,1);
        
    for i = 1:ngraphs

        fprintf('Graph: %s. ',graph_names{i});

        tic;               
        [~,~,atrvertices,edges,~,atredges] =...
            read_grec_gxl(fullfile(dir_graphs,graph_names{i}));
        
        [~,classes_vertices] = ismember(atrvertices,cntrs_atrvertices);
        
        atredges = cellfun(@(x) [x{2},x{4}], atredges,'UniformOutput',false);
        
        [~,classes_edges] = ismember(atredges,cntrs_atredges);
        
        clear atrvertices atredges;

        graphlets = generate_random_graphlets(uint32(edges),MAX1,MAX2);
        
        ngraphlets = size(graphlets,1);
        
        sizes_graphlets = cellfun(@length,graphlets)/2;

        idxle4 = sizes_graphlets<=4;

        graphletsle4 = graphlets(idxle4);

        graphletsle4_sorted = cellfun(@sort,graphletsle4,'UniformOutput',false); clear graphletsge4;
        fcn1 = @(x) sort(diff([ 0 find(x(1:end-1) ~= x(2:end)) length(x) ]));
        sorted_degrees_nodes = cellfun(fcn1,graphletsle4_sorted,'UniformOutput',false);

        clear graphletsle4 graphletsle4_sorted;

        idxge5 = sizes_graphlets>=5;

        graphletsge5 = graphlets(idxge5);

        [list_vertices,~,indices_vertices] = cellfun(@unique,graphletsge5,'UniformOutput',false);                        
        betweenness_centralities = cell(size(graphletsge5));            
        szA = cellfun(@length,list_vertices);

        for j = 1:size(graphletsge5,1)

            A = sparse(indices_vertices{j}(1:2:end),indices_vertices{j}(2:2:end),1,szA(j),szA(j));
            betweenness_centralities{j} = sort(betweenness_centrality(double(A|A')))';

        end;

        clear graphletsge5;

        hash_codes = cell(size(graphlets));

        hash_codes(idxle4) = sorted_degrees_nodes;
        hash_codes(idxge5) = betweenness_centralities;

        % calculate node signatures with the help of classes of the vertices
        node_sign = cellfun(@(x) sort(classes_vertices(x))',graphlets,'UniformOutput',false);
        
        % calculate edge signatures with the help of classes of the vertices
        edges_graphlets = cellfun(@(x) [x(1:2:end)' x(2:2:end)'],graphlets,'UniformOutput',false);
        [~,idx_edges] = cellfun(@(x) ismember(x,edges,'rows'), edges_graphlets, 'UniformOutput', false);
        edge_sign = cellfun(@(x) classes_edges(x)', idx_edges, 'UniformOutput', false);

        for j = 1:size(hash_codes,1)
            hash_codes{j} = [node_sign{j},edge_sign{j},hash_codes{j},zeros(1,2*sizes_graphlets(j)-size(hash_codes{j},2))];
        end;
        
%         coors_graphlets = cell2mat(cellfun(@(x) mean(G.V(x,:)),graphlets,'UniformOutput',false));

        clear idxle4 idxge5 sorted_degrees_nodes betweenness_centralities;

        for j = 1:MAX2
            idxj = (sizes_graphlets == j);
            
            if(~nnz(idxj))
                continue;
            end;
            
            hash_codes_j = cat(1,hash_codes{idxj});
            hash_codes_uniq{j} = unique([hash_codes_uniq{j};hash_codes_j],'stable','rows');
            [~,idx] = ismember(hash_codes_j,hash_codes_uniq{j},'rows');

            clear hash_codes_j;

            idx_image{j} = [idx_image{j};i*ones(size(idx))];
            idx_bin{j} = [idx_bin{j};idx];
%             coors_feats{j} = [coors_feats{j};coors_graphlets(idxj,:)];
            
            clear idx idxj;
        end;

        toc;

    end;

    [dim_hists,~] = cellfun(@size,hash_codes_uniq);% Need to correct it here.

    clear hash_codes_uniq;

    save(file_hist_indices,'idx_image','idx_bin','dim_hists');
    
else
    
    load(file_hist_indices,'idx_image','idx_bin','dim_hists');

end;

tot_time = toc(temp);

%% Compute histograms and kernels

if(~exist(file_kernels,'file')||force_compute.kernels)

    histograms = cell(MAX2,1);

    KM_train = zeros(ntrain_set,ntrain_set,MAX2);
    KM_valid = zeros(nvalid_set,ntrain_set,MAX2);
    
    for i = 1:MAX2

        fprintf('Computing %dth histograms and kernel...',i);

        histograms{i} = sparse(idx_image{i},idx_bin{i},1,ngraphs,dim_hists(i));
        histograms{i} = bsxfun(@times, histograms{i},1./(sum(histograms{i},2)+eps));

        X_train = histograms{i}(train_set,:);
        X_valid = histograms{i}(valid_set,:);

        KM_train(:,:,i) = vl_alldist2(X_train',X_train','KL1');
        KM_valid(:,:,i) = vl_alldist2(X_valid',X_train','KL1');

        fprintf('Done.\n');

    end;
    
    save(file_kernels,'KM_train','KM_valid');    
    clear idx_image idx_bin dim_hists coors_feats;
    
else
    
    load(file_kernels,'KM_train','KM_valid');
    clear idx_image idx_bin dim_hists coors_feats;
    
end;
%% Training and testing individual kernels, multi-class classifier

acc = zeros(1,MAX2+1);
                
for i = 1:MAX2

    K_train = [(1:ntrain_set)' KM_train(:,:,i)];
    K_valid = [(1:nvalid_set)' KM_valid(:,:,i)];

    best_cv = 0;
    best_C = 0;

    for j=1:length(cc)

        options = sprintf('-s 0 -t 4 -g 0.07 -c %.2f -h 0 -b 1 %s-v 5 -q 1',cc(j),w_str);
        model_libsvm = svmtrain(train_classes,K_train,options);

        if(model_libsvm>best_cv)
            best_cv = model_libsvm;
            best_C = cc(j);
        end;

    end;

    options = sprintf('-s 0 -t 4 %s-c %f -b 1 -g 0.07 -h 0 -q',w_str,best_C);

    model_libsvm = svmtrain(train_classes,K_train,options);

    [~,acc_,~] = svmpredict(valid_classes,K_valid,model_libsvm,'-b 0');
    acc(i) = acc_(1);

    fp = fopen(file_results,'a');

    fprintf(fp,'MAX1 = %d, %dth Acc. = %.2f\n\n',MAX1,i,acc(i));

    fclose(fp);

    fprintf('MAX1 = %d, %dth Acc. = %.2f\n\n',MAX1,i,acc(i));

end;
%% Training and testing combined kernels, multi-class classifier
                
w = ones(1,MAX2);
w = w/sum(MAX2);

K_train = [(1:ntrain_set)' lincomb(KM_train,w)];
K_valid = [(1:nvalid_set)' lincomb(KM_valid,w)];

best_cv = 0;
best_C = 0;

for j=1:length(cc)

    options = sprintf('-s 0 -t 4 -g 0.07 -c %.2f -h 0 -b 1 %s-v 5 -q 1',cc(j),w_str);
    model_libsvm = svmtrain(train_classes,K_train,options);

    if(model_libsvm>best_cv)
        best_cv = model_libsvm;
        best_C = cc(j);
    end;

end;

options = sprintf('-s 0 -t 4 %s-c %f -b 1 -g 0.07 -h 0 -q',w_str,best_C);

model_libsvm = svmtrain(train_classes,K_train,options);

[~,acc_,~] = svmpredict(valid_classes,K_valid,model_libsvm,'-b 0');
acc(MAX2+1) = acc_(1);

fp = fopen(file_results,'a');

fprintf(fp,'MAX1 = %d, MAX2 = %d, Comb. Acc. = %.2f\n\n',MAX1,MAX2,acc(MAX2+1));

fclose(fp);

fprintf('MAX1 = %d, MAX2 = %d, Comb. Acc. = %.2f\n\n',MAX1,MAX2,acc(MAX2+1));

%% Get which kernel gives maximum accuracy and compute the diss mat and print 

acc = round2(acc,1e-4);
idx = find(acc == max(acc),1,'last');

if(idx<=MAX2)
    DM = vl_alldist2(histograms{idx}',histograms{idx}','L1');
else
    histogram = cat(2,histograms{:});
    histogram = bsxfun(@times, histogram,1./(sum(histogram,2)+eps));
    DM = vl_alldist2(histogram',histogram','L1');
end;

dlmwrite(file_diss_matrix,DM,'delimiter',' ');
fp = fopen(file_time,'w');
fprintf(fp,'%.2f secs.',tot_time);
fclose(fp);