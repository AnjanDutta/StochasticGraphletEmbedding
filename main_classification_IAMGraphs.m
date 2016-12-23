function main_classification_IAMGraphs(database)

path_dependency;

user = getenv('USER');

switch user
    case 'adutta'
        workhome = '/data/';
    case 'dag'
        workhome = '/home/';
end;

IAMGraphsroot = [workhome,user,'/Workspace/Classification/Datasets/IAMGraphs/'];

file_angle_vocab = fullfile(IAMGraphsroot,'SavedData',database,'angle_vocab.mat');
file_hist_indices = fullfile(IAMGraphsroot,'SavedData',database,'hist_indices.mat');
file_kernels = fullfile(IAMGraphsroot,'SavedData',database,'kernels.mat');
file_results = fullfile(IAMGraphsroot,'SavedData',database,'results.txt');

force_compute.angle_vocab = true;
force_compute.hist_indices = true;
force_compute.kernels = true;

MAX1 = uint32(1000);
MAX2 = uint32(3);

%% Prepare training and test set 

switch database
    case {'COIL-DEL','COIL-DELsmall'}        
        train_file_names = sort([read_cxl(fullfile(IAMGraphsroot,database,'data','train.cxl'));read_cxl(fullfile(IAMGraphsroot,database,'data','valid.cxl'))]);
        test_file_names = read_cxl(fullfile(IAMGraphsroot,database,'data','test.cxl'));
        
        file_names = [train_file_names;test_file_names];
        
        clss = str2double(cellfun(@(x) x{1},regexp(file_names,'obj([0-9])*__\d*.gxl','tokens')));
    case 'GRECGraph'
        clss = str2double(cellfun(@(x) x{1},regexp({graph_names(1:end).name},'image([0-9])*_\d*.gxl','tokens')))';
end;

classes = unique(clss);
nclasses = size(classes,1);

ntrain_set = size(train_file_names,1);
ntest_set = size(test_file_names,1);
ngraphs = size(file_names,1);

train_set = 1:ntrain_set;
test_set = ntrain_set+1:ngraphs;

train_classes = clss(train_set);
test_classes = clss(test_set);

angle_edges = 0:30:180;
histogram_type = 'sp';

%% Create an angle vocabulary

if(~exist(file_angle_vocab,'file')||force_compute.angle_vocab)
    
    num_cls = 10;
    angle_hists = zeros(ngraphs*MAX1*MAX2,length(angle_edges));
    
    fprintf('Creating angle vocabulary...');
        
    for i = 1:ngraphs

        switch database
            case {'COIL-DEL','COIL-DELsmall'}
                G = read_COILDEL_gxl(fullfile(IAMGraphsroot,database,'data',file_names{i}));
            case 'GRECGraph'
                G = read_GREC_gxl(fullfile(IAMGraphsroot,database,'data',file_names{i}));
        end;

        [I,J] = find(G.E);

        L = uint32([I,J]);

        clear I J;

        graphlets = generate_random_graphlets(L,MAX1,MAX2);
        
        ngraphlets = size(graphlets,1);
        
        for j = 1:ngraphlets
            angle_hists((i-1)*MAX1*MAX2+j,:) = histc(cal_angles_horizon([G.V(graphlets{j}(1:2:end)',:),G.V(graphlets{j}(2:2:end)',:)]),angle_edges);
        end;
        
    end;
    
    angle_hists = bsxfun(@times,angle_hists,1./(sum(angle_hists,2)+eps));
    [~,cntrs_angle] = kmeans(angle_hists,num_cls,'EmptyAction','singleton');
    clear angle_hists;
    
    save(file_angle_vocab,'cntrs_angle');
   
    fprintf('Done.\n');
    
else
    
    fprintf('Angle vocabulary already exist.\n');    
    load(file_angle_vocab,'cntrs_angle');
    
end;

%% Now create histogram indices

if(~exist(file_hist_indices,'file')||force_compute.hist_indices)
    
    hash_codes_uniq = cell(MAX2,1);
    idx_image = cell(MAX2,1);
    idx_bin = cell(MAX2,1);
    coors_feats = cell(MAX2,1);
        
    for i = 1:ngraphs

        fprintf('Graph: %s. ',file_names{i});

        tic;

        switch database
            case {'COIL-DEL','COIL-DELsmall'}
                G = read_COILDEL_gxl(fullfile(IAMGraphsroot,database,'data',file_names{i}));
            case 'GRECGraph'
                G = read_GREC_gxl(fullfile(IAMGraphsroot,database,'data',file_names{i}));
        end;

        [I,J] = find(G.E);

        L = uint32([I,J]);

        clear I J;

        graphlets = generate_random_graphlets(L,MAX1,MAX2);
        
        ngraphlets = size(graphlets,1);
        
        angle_hists = zeros(ngraphlets,length(angle_edges));        
        for j = 1:ngraphlets
            angle_hists(j,:) = histc(cal_angles_horizon([G.V(graphlets{j}(1:2:end)',:),G.V(graphlets{j}(2:2:end)',:)]),angle_edges);
        end;        
        angle_hists = bsxfun(@times,angle_hists,1./(sum(angle_hists,2)+eps));        
        angle_clss = knnsearch(cntrs_angle,angle_hists);
        
        clear angle_hists;
        
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

        for j = 1:size(hash_codes,1)
            hash_codes{j} = [angle_clss(j),hash_codes{j},zeros(1,2*sizes_graphlets(j)-size(hash_codes{j},2))];
        end;
        
        coors_graphlets = cell2mat(cellfun(@(x) mean(G.V(x,:)),graphlets,'UniformOutput',false));

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
            coors_feats{j} = [coors_feats{j};coors_graphlets(idxj,:)];
            
            clear idx idxj;
        end;

        toc;

    end;

    [dim_hists,~] = cellfun(@size,hash_codes_uniq);% Need to correct it here.

    clear hash_codes_uniq;

    save(file_hist_indices,'idx_image','idx_bin','coors_feats','dim_hists');
    
else
    
    load(file_hist_indices,'idx_image','idx_bin','coors_feats','dim_hists');

end;

%% Compute histograms and kernels

if(~exist(file_kernels,'file')||force_compute.kernels)

    histograms = cell(MAX2,1);

    KM_train = zeros(ntrain_set,ntrain_set,MAX2);
    KM_test = zeros(ntest_set,ntrain_set,MAX2);
    
    switch histogram_type
        
        case 'normal'            

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
    
        case 'sp'
            
            numSpatialX = 5;
            numSpatialY = 5;
            
            hists = cell(size(numSpatialX));

            mult_factor_spm = sum(numSpatialX.*numSpatialY);
            
            for k = 1:MAX2
                histograms{k} = zeros(ngraphs,dim_hists(k)*mult_factor_spm);
            end;
            
            for j = 1:ngraphs
                
                fprintf('Computing spatial histogram for %05dth image.\n',j);
                
                for i = 1:MAX2

                    idx = (idx_image{i} == j);
                    
                    dim = max(coors_feats{i})+1;

                    pos = double(coors_feats{i}(idx,:));
                    bin = idx_bin{i}(idx,:);                    

                    for ix = 1:length(numSpatialX)

                        binsx = vl_binsearch(linspace(1,dim(1),numSpatialX(ix)+1),pos(:,1)');
                        binsy = vl_binsearch(linspace(1,dim(2),numSpatialY(ix)+1),pos(:,2)');

                        bins = sub2ind([numSpatialY(ix),numSpatialX(ix),dim_hists(i)],binsy,binsx,bin');

                        hist = zeros(numSpatialY(ix)*numSpatialX(ix)*dim_hists(i),1);
                        hist = vl_binsum(hist,ones(size(bins)),bins) ;
                        hists{ix} = single(hist/sum(hist)) ;

                    end;

                    hist = cat(1,hists{:}) ;
                    hist = hist / sum(hist) ;

                    histograms{i}(j,:) = hist';
                end;
            end;            
            
            for i = 1:MAX2             
                
                fprintf('Computing %02dth kernel...',i);

                X_train = histograms{i}(train_set,:);
                X_test = histograms{i}(test_set,:);

                KM_train(:,:,i) = vl_alldist2(X_train',X_train','KL1');
                KM_test(:,:,i) = vl_alldist2(X_test',X_train','KL1');
                
                fprintf('Done.\n');

            end;
            
    end;

    save(file_kernels,'KM_train','KM_test');    
    clear idx_image idx_bin dim_hists coors_feats;
    
else
    
    load(file_kernels,'KM_train','KM_test');
    clear idx_image idx_bin dim_hists coors_feats;
    
end;



%% Training and testing individual kernels, multi-class classifier
                
for i = 1:MAX2

    K_train = [(1:ntrain_set)' KM_train(:,:,i)];
    K_test = [(1:ntest_set)' KM_test(:,:,i)];

    best_C = 10.0;

    options = sprintf('-s 0 -t 4 -c %f -b 1 -g 0.07 -h 0 -q',best_C);

    model_libsvm = svmtrain(train_classes,K_train,options);

    [cls_pred,~,~] = svmpredict(test_classes,K_test,model_libsvm,'-b 1');

    acc = nnz(test_classes == cls_pred)/length(test_classes);

    fp = fopen(file_results,'a');

    fprintf(fp,'%dth Acc = %.2f\t',i,mean(acc)*100);

    fclose(fp);

    fprintf('Accuracy = %.2f\n\n',mean(acc)*100);

end;
%% Training and testing combined kernels, multi-class classifier
                
w = ones(1,MAX2);
w = w/sum(MAX2);

K_train = [(1:ntrain_set)' lincomb(KM_train,w)];
K_test = [(1:ntest_set)' lincomb(KM_test,w)];

best_C = 10.0;

options = sprintf('-s 0 -t 4 -c %f -b 1 -g 0.07 -h 0 -q',best_C);

model_libsvm = svmtrain(train_classes,K_train,options);

[cls_pred,~,~] = svmpredict(test_classes,K_test,model_libsvm,'-b 1');

acc = nnz(test_classes == cls_pred)/length(test_classes);

fp = fopen(file_results,'a');

fprintf(fp,'Comb. Acc = %.2f\t',mean(acc)*100);

fclose(fp);

fprintf('Accuracy = %.2f\n\n',mean(acc)*100);

%% Training and testing MKL

%{
addpath('/data/adutta/Dropbox/Personal/Workspace/AdditionalTools/GMKL');

confs = zeros(ntest_set,nclasses);

for iclass = 1:nclasses
    
    cls = classes(iclass);

    fprintf('Class = %d\n',cls);

    Y_train = -ones(ntrain_set,1);
    Y_test = -ones(ntest_set,1);

    Y_train(train_classes==cls) = 1;
    Y_test(test_classes==cls) = 1;

    npos_train = nnz(Y_train == 1);
    nneg_train = nnz(Y_train == -1);
    npos_test = nnz(Y_test == 1);
    nneg_test = nnz(Y_test == -1);

    display(['TrainSet: number of positives = ',num2str(npos_train),', number of negatives = ',num2str(nneg_train)]);
    display(['TestSet: number of positives = ',num2str(npos_test),', number of negatives = ',num2str(nneg_test)]);

    fprintf('Learning the weights of GMKL...');

    gmklsvm = learnGmklSvm(KM_train,Y_train);
    gmklsvm = svmflip(gmklsvm,Y_train);

    fprintf('Done.\n'); 

    % test it on train set
    scores =  gmklsvm.alphay'*lincomb(KM_train(gmklsvm.svind,:,:),gmklsvm.d) + gmklsvm.b ;

    errs = scores' .* Y_train < 0 ;
    err  = mean(errs) ;
    selPos = find(Y_train > 0) ;
    selNeg = find(Y_train < 0) ;
    werr = sum(errs(selPos)) + sum(errs(selNeg)) ;
    werr = werr / (length(selPos) + length(selNeg)) ;
    fprintf('SVM training error: %.2f%% (weighed: %.2f%%).\n', ...
          err*100, werr*100) ;

    % test it on test set
    conf = lincomb(KM_test(:,gmklsvm.svind,:),gmklsvm.d)*gmklsvm.alphay + gmklsvm.b;

    confs(:,cls) = conf;

end;

[~,cls_pred] = max(confs,[],2);

acc = nnz(test_classes == cls_pred)/length(test_classes);

fp = fopen(file_results,'a');

fprintf(fp,'MKL Acc = %.2f\t',mean(acc)*100);

fclose(fp);

fprintf('Accuracy = %.2f\n\n',mean(acc)*100);
%}