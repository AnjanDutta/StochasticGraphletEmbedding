user = getenv('USER');

switch user
    case 'root'
        workhome = '/home/anjan/';
    case 'dag'
        workhome = '/home/dag/';
end;

addpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/random_graphlet1']);
addpath(genpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/vlfeat']));
addpath(genpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/matlab_bgl']));
addpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/libsvm/matlab']);
addpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/support']);
rmpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/vlfeat/toolbox/kmeans']);
rmpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/vlfeat/toolbox/gmm']);
rmpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/vlfeat/toolbox/fisher']);
rmpath([workhome,'Dropbox/Personal/Workspace/AdditionalTools/vlfeat/toolbox/noprefix']);
