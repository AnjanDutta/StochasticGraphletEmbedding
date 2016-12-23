function main_classification_mao()

path_dependency;

user = getenv('USER');

switch user
    case 'root'
        workhome = '/home/anjan/Workspace/';
    case 'dag'
        workhome = '/home/dag/Workspace/';
    case 'adutta'
        workhome = '/home/dag/adutta/';
end;

file_node_vocab = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/MAO','node_vocab.mat');
file_edge_vocab = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/MAO','edge_vocab.mat');

force_compute.vocab = false;
force_compute.hist_indices = false;
force_compute.kernels = true;

cc = [1 10 50 100 1000];
%% Prepare training and test set

% seeding randomization
rng(0);

dir_graphs = [workhome,'Datasets/GREYCGraphs/MAO/'];

fp = fopen(fullfile(dir_graphs,'dataset.ds')) ;
C = textscan(fp, '%s %d') ;
fclose(fp) ;

graph_names = C{1} ;
classes = double(C{2}) ;
clear C ;
classes(~classes) = 2 ;
uniq_classes = unique(classes)' ;

niter = 10 ;
train_set = cell(niter, 1) ;
test_set = cell(niter, 1) ;
for iter = 1:niter
    for ic = uniq_classes
        idx = find(classes == ic) ;
        train_set{iter} = [train_set{iter};randsample(idx,round(0.5*length(idx)))] ;
        test_set{iter} = [test_set{iter};setdiff(idx,train_set{iter})] ;
    end;
    train_set{iter} = sort(train_set{iter}) ;
    test_set{iter} = sort(test_set{iter}) ;
end;

ngraphs = size(graph_names,1);
w_str = [];
for ic = uniq_classes
    nc = nnz(classes == ic);
    w_str = [w_str,sprintf('-w%d %.2f ',ic,ngraphs/nc)];
end;

%% First create a node and edge vocabulary

if(~exist(file_node_vocab,'file')||force_compute.vocab)
    
    cntrs_atrvertices = [];
        
    fprintf('Creating node vocabulary...');
    
    for i = 1:ngraphs
        
        [~,~,atrvertices,~,~,~] =...
            read_mao_ct(fullfile(dir_graphs,graph_names{i}));
              
        cntrs_atrvertices = unique([cntrs_atrvertices;atrvertices]);
               
        clear atrvertices;
        
    end;
    
    save(file_node_vocab,'cntrs_atrvertices');    
       
    fprintf('Done.\n');
    
else
    
    fprintf('Node vocabulary already exist.\n');    
    
    load(file_node_vocab,'cntrs_atrvertices');
        
end;
%% Now create histogram indices

epss = [0.1 0.05];
dels = [0.1 0.05];

MAX2 = uint32(5);

T = [1 1 3 5 12 30 79 227 710 2322 8071];

for ieps = 1:2
    for idel = 1:2
        
        file_hist_indices = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/MAO',sprintf('hist_indices_%d_%d.mat',ieps,idel));
        file_kernels = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/MAO',sprintf('kernels_%d_%d.mat',ieps,idel));
        file_results = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/MAO',sprintf('results_%d_%d.txt',ieps,idel));
        
        epsi = epss(ieps);
        deli = dels(idel);
        
        MAX1 = uint32(ceil(2*(T(1:MAX2)*log(2)+log(1/deli))/epsi^2));
        
        if(~exist(file_hist_indices,'file')||force_compute.hist_indices)

            hash_codes_uniq = cell(MAX2,1);
            idx_image = cell(MAX2,1);
            idx_bin = cell(MAX2,1);

            for i = 1:ngraphs

                fprintf('Graph: %s. ',graph_names{i});

                tic;

                [~,~,atrvertices,edges,~,atredges] =...
                    read_mao_ct(fullfile(dir_graphs,graph_names{i}));
                
                [~,classes_vertices] = ismember(atrvertices,cntrs_atrvertices);

                classes_edges1 = atredges(:,1) ;
                
                classes_edges2 = atredges(:,2) ;

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
                edge_sign1 = cellfun(@(x) classes_edges1(x)', idx_edges, 'UniformOutput', false);
                edge_sign2 = cellfun(@(x) classes_edges2(x)', idx_edges, 'UniformOutput', false);

                clear edges_graphlets idx_edges;

                for j = 1:size(hash_codes,1)
                    hash_codes{j} = [node_sign{j},edge_sign1{j},edge_sign2{j},hash_codes{j},zeros(1,2*sizes_graphlets(j)-size(hash_codes{j},2))];
                end;

                clear idxle4 idxge5 sorted_degrees_nodes betweenness_centralities node_sign edge_sign;

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

        %% Compute histograms and kernels
        
        acc = zeros(niter,MAX2+1);
        
        for iter = 1:niter
            
            train_seti = train_set{iter} ;
            test_seti = test_set{iter} ;
            
            ntrain_set = size(train_seti,1) ;
            ntest_set = size(test_seti,1) ;
            
            train_classes = classes(train_seti,:) ;
            test_classes = classes(test_seti,:) ;

            if(~exist(file_kernels,'file')||force_compute.kernels)

                histograms = cell(MAX2,1);

                KM_train = zeros(ntrain_set,ntrain_set,MAX2);
                KM_test = zeros(ntest_set,ntrain_set,MAX2);

                for i = 1:MAX2

                    fprintf('Computing %dth histograms and kernel...',i);

                    histograms{i} = sparse(idx_image{i},idx_bin{i},1,ngraphs,dim_hists(i));
                    histograms{i} = bsxfun(@times, histograms{i},1./(sum(histograms{i},2)+eps));

                    X_train = histograms{i}(train_seti,:);
                    X_test = histograms{i}(test_seti,:);

                    KM_train(:,:,i) = vl_alldist2(X_train',X_train','KL1');
                    KM_test(:,:,i) = vl_alldist2(X_test',X_train','KL1');

                    fprintf('Done.\n');

                end;

                save(file_kernels,'KM_train','KM_test');    
%                 clear idx_image idx_bin dim_hists coors_feats;

            else

                load(file_kernels,'KM_train','KM_test');
                clear idx_image idx_bin dim_hists coors_feats;

            end;
            %% Training and testing individual kernels, multi-class classifier

            for i = 1:MAX2

                K_train = [(1:ntrain_set)' KM_train(:,:,i)];
                K_test = [(1:ntest_set)' KM_test(:,:,i)];

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

                options = sprintf('-s 0 -t 4 -q 1 %s-c %f -b 1 -g 0.07 -h 0',w_str,best_C);

                model_libsvm = svmtrain(train_classes,K_train,options);

                [~,acc_,~] = svmpredict(test_classes,K_test,model_libsvm,'-b 0');
                acc(iter,i) = acc_(1);

                fp = fopen(file_results,'a');

                fprintf(fp,'MAX1 = %d, %dth Acc. = %.2f\n\n',MAX1(i),i,acc(i));

                fclose(fp);

                fprintf('MAX1 = %d, %dth Acc. = %.2f\n\n',MAX1(i),i,acc(i));

            end;
            %% Training and testing combined kernels, multi-class classifier

            w = ones(1,MAX2);
            w = w/sum(MAX2);

            K_train = [(1:ntrain_set)' lincomb(KM_train,w)];
            K_test = [(1:ntest_set)' lincomb(KM_test,w)];

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

            options = sprintf('-s 0 -t 4 -q %s-c %f -b 1 -g 0.07 -h 0',w_str,best_C);

            model_libsvm = svmtrain(train_classes,K_train,options);

            [~,acc_,~] = svmpredict(test_classes,K_test,model_libsvm,'-b 1');
            acc(iter,MAX2+1) = acc_(1);

            fp = fopen(file_results,'a');

            fprintf(fp,'MAX1 = %d, MAX2 = %d, Comb. Acc. = %.2f\n\n',MAX1(end),MAX2,acc(MAX2+1));

            fclose(fp);

            fprintf('MAX1 = %d, MAX2 = %d, Comb. Acc. = %.2f\n\n',MAX1(end),MAX2,acc(MAX2+1));
            
        end;
        
        disp(mean(acc)) ;
        
    end;
end;
%% Get which kernel gives maximum accuracy and compute the diss mat and print 

% acc = round2(acc,1e-4);
% idx = find(acc == max(acc),1,'last');
% 
% if(idx<=MAX2)
%     DM = vl_alldist2(histograms{idx}',histograms{idx}','L1');
% else
%     histogram = cat(2,histograms{:});
%     histogram = bsxfun(@times, histogram,1./(sum(histogram,2)+eps));
%     DM = vl_alldist2(histogram',histogram','L1');
% end;
% 
% dlmwrite(file_diss_matrix,DM,'delimiter',' ');
% fp = fopen(file_time,'w');
% fprintf(fp,'%.2f secs.',tot_time);
% fclose(fp);