%% example of training and testing OSVR for expression intensity estimation
clear all; close all;

%% load data
% train_data_seq: an array of cells containing training feature sequences,
% each cell contains a D*T matrix where D is dimension of feature and T is
% the sequence length
% train_label_seq: an array of cells containing training intensity labels
% for all the sequences, each cell contains a K*2 matrix where K is the
% number of frames with labeled intensities. The first column is the index
% of frame and the second column is associated intensity value
% test_data: a D*T' matrix containing testing frames, where D is the
% dimension of feature and T' is number of testing frames
% data preparation
load('feature_from_verification_model.mat');
%load('data.mat','train_data_seq','train_label_seq','test_data','test_label');
num_ppl = length(video_feature);
% construct training data
vid_count = 0;
for i=2:num_ppl
    idx_list = video_index{i,1};
    numVid = length(idx_list);
    for j=1:numVid
        vid_count = vid_count + 1;
        num_frm = idx_list(j);
        train_data_seq{1,vid_count} = video_feature{i,1}{j,1};
        train_label_seq{1,vid_count}(:,1) = 1:num_frm;
        train_label_seq{1,vid_count}(:,2) = video_pain_level{i,1}{j,1};
    end
end
% construct testing data
%idx_list = video_index{1,1};
%numVid = length(idx_list);
test_data = video_feature{1,1}{1,1};
test_label = video_pain_level{1,1}{1,1};

%% define constant
loss = 2; % loss function of OSVR
bias = 1; % include bias term or not in OSVR
lambda = 1; % scaling parameter for primal variables in OSVR
gamma = [100 1]; % loss balance parameter
smooth = 1; % temporal smoothness on ordinal constraints
epsilon = [0.1 1]; % parameter in epsilon-SVR
rho = 0.1; % augmented Lagrangian multiplier
flag = 0; % unsupervise learning flag
max_iter = 300; % maximum number of iteration in optimizating OSVR

%% Training 
% formalize coefficients data structure
[A,c,D,nInts,nPairs,weight] = constructParams(train_data_seq,train_label_seq,epsilon,bias,flag);
mu = gamma(1)*ones(nInts+nPairs,1); % change the values if you want to assign different weights to different samples
mu(nInts+1:end) = gamma(2)/gamma(1)*mu(nInts+1:end);
if smooth % add temporal smoothness
    mu = mu.*weight;
end
% solve the OSVR optimization problem in ADMM
[model,history,z] = admm(A,c,lambda,mu,'option',loss,'rho',rho,'max_iter',max_iter,'bias',1-bias); % 
theta = model.w;
     
%% Testing 
% perform testing
dec_values =theta'*[test_data; ones(1,size(test_data,2))];
% compute evaluation metrics
RR = corrcoef(dec_values,test_label);  
ee = dec_values - test_label; 
dat = [dec_values; test_label]'; 
ry_test = RR(1,2); % Pearson Correlation Coefficient (PCC)
abs_test = sum(abs(ee))/length(ee); % Mean Absolute Error (MAE)
mse_test = ee(:)'*ee(:)/length(ee); % Mean Square Error (MSE)
icc_test = ICC(3,'single',dat); % Intra-Class Correlation (ICC)

% adjust 1: lift estimated min to 0 due to non-negativity
dec_values_plus = dec_values - min(dec_values);

% adjust 2: only lift the negative part
% dec_values_adjust = dec_values;
% for i=1:length(dec_values)
%     if dec_values_adjust(i)<0
%         dec_values_adjust(i) = 0;
%     end
% end

%% Visualize results
plot(test_label); hold on; 
plot(dec_values_plus,'r');
legend('Ground truth','Prediction')

% just lift 1 level for the top 200 frames
% for i=1:200
%     dec_values_adjust(i) = dec_values_adjust(i) + 1;
% end