%% Neural Network – Fixed Architecture, Enhanced Inputs
% Hidden layers: 256 → 128 → 64 (GELU)
% Inputs: 20 lagged flow rates + time index

clear; clc; close all;

fprintf('========================================\n');
fprintf('FPGA-FRIENDLY NETWORK (256-128-64)\n');
fprintf('Enhanced with lagged features & time index\n');
fprintf('========================================\n\n');

%% 1. Load Data
filename = 'myData.csv';          % <-- CHANGE TO YOUR FILE
headerLines = 1;
data = readmatrix(filename, 'NumHeaderLines', headerLines);
flow_rate = data(:, 2);            % adjust column index if needed
angular_velocity = data(:, 3);     % adjust column index if needed

X_raw = flow_rate;
T_raw = angular_velocity;

%% 2. Create Lagged Features (history)
lag = 20;   % number of past samples (tune: 10-30)
X_lagged = [];
for i = 1:lag
    X_lagged = [X_lagged, circshift(X_raw, i)];
end
% Remove rows with NaN (first lag rows)
X_lagged = X_lagged(lag+1:end, :);
T_raw = T_raw(lag+1:end);

% Add normalized time index as an additional feature
time_idx = (1:length(T_raw))';
time_idx_norm = (time_idx - mean(time_idx)) / std(time_idx);
X_features = [X_lagged, time_idx_norm];   % size N x (lag+1)

fprintf('Input features: %d (lag=%d + time index)\n', size(X_features,2), lag);
fprintf('Samples after lagging: %d\n', length(T_raw));

%% 3. Normalization (standardize each input feature)
X_mean = mean(X_features);
X_std  = std(X_features);
T_mean = mean(T_raw);
T_std  = std(T_raw);

X = (X_features - X_mean) ./ X_std;
T = (T_raw - T_mean) ./ T_std;

%% 4. Split Data (70/15/15)
numSamples = size(X,1);
indices = randperm(numSamples);
trainEnd = floor(0.70 * numSamples);
valEnd   = trainEnd + floor(0.15 * numSamples);

trainIdx = indices(1:trainEnd);
valIdx   = indices(trainEnd+1:valEnd);
testIdx  = indices(valEnd+1:end);

XTrain = X(trainIdx,:);
TTrain = T(trainIdx,:);
XVal   = X(valIdx,:);
TVal   = T(valIdx,:);
XTest  = X(testIdx,:);
TTest  = T(testIdx,:);

fprintf('Data split: Train=%d, Val=%d, Test=%d\n', ...
    length(trainIdx), length(valIdx), length(testIdx));

%% 5. Network Definition (256-128-64, GELU)
layers = [
    featureInputLayer(size(X,2), 'Normalization', 'none', 'Name', 'input')
    fullyConnectedLayer(256, 'Name', 'fc1')
    geluLayer('Name', 'gelu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    geluLayer('Name', 'gelu2')
    fullyConnectedLayer(64, 'Name', 'fc3')
    geluLayer('Name', 'gelu3')
    fullyConnectedLayer(1, 'Name', 'fc_out')
    regressionLayer('Name', 'output')
];

%% 6. Training Options (SGDM with cosine annealing)
if exist('trainingOptions', 'file') && ...
   isprop(trainingOptions('sgdm'), 'LearnRateSchedule') && ...
   strcmp(trainingOptions('sgdm').LearnRateSchedule, 'cosine')
    % Use cosine annealing if available (R2022b+)
    options = trainingOptions('sgdm', ...
        'MaxEpochs', 800, ...
        'MiniBatchSize', 1024, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'cosine', ...
        'Momentum', 0.9, ...
        'L2Regularization', 0.0001, ...
        'ValidationData', {XVal, TVal}, ...
        'ValidationFrequency', 30, ...
        'Verbose', true, ...
        'VerboseFrequency', 30, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch', ...
        'GradientThreshold', 1);
else
    % Fallback to piecewise schedule
    options = trainingOptions('sgdm', ...
        'MaxEpochs', 800, ...
        'MiniBatchSize', 1024, ...
        'InitialLearnRate', 0.01, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 200, ...
        'Momentum', 0.9, ...
        'L2Regularization', 0.0001, ...
        'ValidationData', {XVal, TVal}, ...
        'ValidationFrequency', 30, ...
        'Verbose', true, ...
        'VerboseFrequency', 30, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch', ...
        'GradientThreshold', 1);
end

fprintf('\nTraining Configuration:\n');
fprintf('  Optimizer: SGDM (momentum 0.9)\n');
fprintf('  Max epochs: %d\n', options.MaxEpochs);
fprintf('  L2 regularization: %.4f\n', options.L2Regularization);

%% 7. Train Network
fprintf('\n=== Starting Training ===\n');
tic;
net = trainNetwork(XTrain, TTrain, layers, options);
trainingTime = toc;
fprintf('Training completed in %.2f seconds (%.2f minutes)\n', trainingTime, trainingTime/60);

%% 8. Evaluate Performance
fprintf('\n=== Evaluating Performance ===\n');
Y_pred_scaled = predict(net, X);
Y_pred = Y_pred_scaled * T_std + T_mean;

% Overall metrics
mse_overall = mean((T_raw - Y_pred).^2);
SS_res = sum((T_raw - Y_pred).^2);
SS_tot = sum((T_raw - mean(T_raw)).^2);
R2 = 1 - (SS_res / SS_tot);
fprintf('Overall MSE: %.6f\n', mse_overall);
fprintf('Overall R²: %.4f (%.2f%%)\n', R2, R2*100);

% Per‑set metrics
mse_train = mean((T_raw(trainIdx) - Y_pred(trainIdx)).^2);
mse_val   = mean((T_raw(valIdx)   - Y_pred(valIdx)).^2);
mse_test  = mean((T_raw(testIdx)  - Y_pred(testIdx)).^2);
fprintf('Training MSE: %.6f\n', mse_train);
fprintf('Validation MSE: %.6f\n', mse_val);
fprintf('Test MSE: %.6f\n', mse_test);

%% 9. Visualization (focus on valley)
figure('Position', [100, 100, 1200, 400]);

time = (1:length(T_raw))';
subplot(1,3,1);
plot(time, T_raw, 'b-', 'LineWidth', 1);
hold on;
plot(time, Y_pred, 'r--', 'LineWidth', 1);
xlabel('Sample index');
ylabel('Angular velocity');
title('Time Series: Actual vs Predicted');
legend('Actual', 'Predicted', 'Location', 'best');
grid on;

subplot(1,3,2);
plot(T_raw, Y_pred, 'b.', 'MarkerSize', 2);
hold on;
plot([min(T_raw), max(T_raw)], [min(T_raw), max(T_raw)], 'r-', 'LineWidth', 2);
xlabel('Actual');
ylabel('Predicted');
title(sprintf('Scatter Plot (R² = %.2f%%)', R2*100));
grid on;

subplot(1,3,3);
errors = T_raw - Y_pred;
plot(time, errors, 'k.', 'MarkerSize', 1);
xlabel('Sample index');
ylabel('Error');
title(sprintf('Prediction Error (MSE = %.4f)', mse_overall));
grid on;

% Zoom on valley region (adjust indices based on your data)
valley_start = 40000;
valley_end = 70000;
figure;
plot(time(valley_start:valley_end), T_raw(valley_start:valley_end), 'b-', 'LineWidth', 1.5);
hold on;
plot(time(valley_start:valley_end), Y_pred(valley_start:valley_end), 'r--', 'LineWidth', 1.5);
xlabel('Sample index');
ylabel('Angular velocity');
title('Zoomed Valley Region');
legend('Actual', 'Predicted', 'Location', 'best');
grid on;

%% 10. Extract Weights for FPGA (unchanged from your original)
fc1_weights = net.Layers(2).Weights;  % 256 x input_dim
fc1_bias    = net.Layers(2).Bias;
fc2_weights = net.Layers(4).Weights;  % 128 x 256
fc2_bias    = net.Layers(4).Bias;
fc3_weights = net.Layers(6).Weights;  % 64 x 128
fc3_bias    = net.Layers(6).Bias;
fc_out_weights = net.Layers(8).Weights; % 1 x 64
fc_out_bias    = net.Layers(8).Bias;

fprintf('\nWeights extracted (for FPGA):\n');
fprintf('  FC1: %d x %d\n', size(fc1_weights));
fprintf('  FC2: %d x %d\n', size(fc2_weights));
fprintf('  FC3: %d x %d\n', size(fc3_weights));
fprintf('  Output: %d x %d\n', size(fc_out_weights));

% Save as .mat file
weights_fpga = struct();
weights_fpga.W1 = fc1_weights;
weights_fpga.b1 = fc1_bias;
weights_fpga.W2 = fc2_weights;
weights_fpga.b2 = fc2_bias;
weights_fpga.W3 = fc3_weights;
weights_fpga.b3 = fc3_bias;
weights_fpga.W4 = fc_out_weights;
weights_fpga.b4 = fc_out_bias;
weights_fpga.X_mean = X_mean;
weights_fpga.X_std = X_std;
weights_fpga.T_mean = T_mean;
weights_fpga.T_std = T_std;
save('nn_weights_fpga_fixed.mat', 'weights_fpga');

fprintf('\n=== READY FOR FPGA IMPLEMENTATION ===\n');
