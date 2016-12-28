function main_classification_coildelsmall()

path_dependency;

workhome = '/home/anjan/Workspace/'; 

force_compute.vocab = false;
force_compute.hist_indices = true;
force_compute.kernels = true;

cc = [1 10 50 100 1000];
%% Prepare training and test set

% seeding randomization
rng(0);

dir_graphs = [workhome,'Datasets/IAMGraphs/COIL-DELsmall/data'];

[train_graph_names,train_classes] = read_coildel_cxl(fullfile(dir_graphs,'train.cxl'));
[valid_graph_names,valid_classes] = read_coildel_cxl(fullfile(dir_graphs,'valid.cxl'));
train_graph_names = [train_graph_names; valid_graph_names] ;
train_classes = [train_classes; valid_classes] ;
[test_graph_names,test_classes] = read_coildel_cxl(fullfile(dir_graphs,'test.cxl'));

graph_names = [train_graph_names;test_graph_names];

train_set = 1:length(train_graph_names);
% valid_set = (1:length(valid_graph_names)) + length(train_set);
test_set = (1:length(test_graph_names)) + length(train_graph_names);

ntrain_set = size(train_graph_names,1);
% nvalid_set = size(valid_graph_names,1);
ntest_set = size(test_graph_names,1);
ngraphs = size(graph_names,1);

uniq_classes = unique(train_classes)';
w_str = [];
for ic = uniq_classes
    nc = nnz(train_classes == ic);
    w_str = [w_str,sprintf('-w%d %.2f ',ic,ntrain_set/nc)];
end;

%% Now create histogram indices

epss = [0.1 0.05];
dels = [0.1 0.05];

MAX2 = uint32(7);

T = [1 1 3 5 12 30 79 227 710 2322 8071];

for ieps = 1:2
    for idel = 1:2
        
        file_hist_indices = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/COIL-DELsmall',sprintf('hist_indices_%d_%d.mat',ieps,idel));
        file_kernels = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/COIL-DELsmall',sprintf('kernels_%d_%d.mat',ieps,idel));
        file_results = fullfile(workhome,'StochasticGraphletEmbedding/SavedData/COIL-DELsmall',sprintf('results_%d_%d.txt',ieps,idel));
        
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
                [~,~,~,edges,~,atredges] =...
                    read_coildel_gxl(fullfile(dir_graphs,graph_names{i}));

                classes_edges = atredges ;

                clear atredges;

                graphlets = generate_random_graphlets(uint32(edges),MAX1(end),MAX2);
                
                sizes_graphlets = cellfun(@length,graphlets)/2;
                
                idx = [] ; 
                for ii = 1:min(MAX2,max(sizes_graphlets))
                    idx1 = find(sizes_graphlets == ii) ;
                    idx = [idx; randsample(idx1, min(MAX1(ii), length(idx1)))] ;
                end;
                
                idx = sort(idx) ;
                graphlets = graphlets(idx) ;
                sizes_graphlets = sizes_graphlets(idx) ;

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

                % calculate edge signatures with the help of classes of the vertices
                edges_graphlets = cellfun(@(x) [x(1:2:end)' x(2:2:end)'],graphlets,'UniformOutput',false);
                [~,idx_edges] = cellfun(@(x) ismember(x,edges,'rows'), edges_graphlets, 'UniformOutput', false);
                edge_sign = cellfun(@(x) classes_edges(x)', idx_edges, 'UniformOutput', false);

                for j = 1:size(hash_codes,1)
                    hash_codes{j} = [edge_sign{j},hash_codes{j},zeros(1,2*sizes_graphlets(j)-size(hash_codes{j},2))];
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

%% Compute histograms and kernels

        if(~exist(file_kernels,'file')||force_compute.kernels)

            histograms = cell(MAX2,1);

            KM_train = zeros(ntrain_set,ntrain_set,MAX2);
            KM_test = zeros(ntest_set,ntrain_set,MAX2);

            for i = 1:MAX2

                fprintf('Computing %dth histograms and kernel...',i);

                histograms{i} = sparse(idx_image{i},idx_bin{i},1,ngraphs,dim_hists(i));
                histograms{i} = bsxfun(@times, histograms{i},1./(sum(histograms{i},2)+eps));

                X_train = histograms{i}(train_set,:);
                X_test = histograms{i}(test_set,:);

                KM_train(:,:,i) = vl_alldist2(X_train',X_train','KL1');
                KM_test(:,:,i) = vl_alldist2(X_test',X_train','KL1');

                fprintf('Done.\n');

            end;

            save(file_kernels,'KM_train','KM_test');    
            clear idx_image idx_bin dim_hists coors_feats;

        else

            load(file_kernels,'KM_train','KM_test');
            clear idx_image idx_bin dim_hists coors_feats;

        end;
%% Training and testing individual kernels, multi-class classifier

        acc = zeros(1,MAX2+1);

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

            options = sprintf('-s 0 -t 4 %s-c %f -b 1 -g 0.07 -h 0 -q',w_str,best_C);

            model_libsvm = svmtrain(train_classes,K_train,options);

            [~,acc_,~] = svmpredict(test_classes,K_test,model_libsvm,'-b 1');
            acc(i) = acc_(1);

            fp = fopen(file_results,'a');

            fprintf(fp,'MAX1 = %d, %dth Acc. = %.2f\n\n',MAX1(i),i,acc(i));

            fclose(fp);

            fprintf('MAX1 = %d, %dth Acc. = %.2f\n\n',MAX1(i),i,acc(i));

        end;
%% Training and testing combined kernels, multi-class classifier
                
        w = ones(1,MAX2);
        w = w/sum(MAX2);

        for i = 1:MAX2
            
            K_train = [(1:ntrain_set)' lincomb(KM_train(:,:,1:i),w(1:i))];
            K_test = [(1:ntest_set)' lincomb(KM_test(:,:,1:i),w(1:i))];

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

            [~,acc_,~] = svmpredict(test_classes,K_test,model_libsvm,'-b 1');
            acc(MAX2+i) = acc_(1);

            fp = fopen(file_results,'a');

            fprintf(fp,'Combined: MAX1 = %d, MAX2 = %d, Acc. = %.2f\n\n',sum(MAX1(1:i)),i,acc(MAX2+i));

            fclose(fp);

            fprintf('Combined: MAX1 = %d, MAX2 = %d, Acc. = %.2f\n\n',sum(MAX1(1:i)),i,acc(MAX2+i));
            
        end;
        
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