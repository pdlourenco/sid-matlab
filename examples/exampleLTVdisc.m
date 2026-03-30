%% exampleLTVdisc - LTV state-space identification with sidLTVdisc
%
% This example demonstrates sidLTVdisc (COSMIC algorithm) for identifying
% time-varying discrete linear systems of the form:
%
%   x(k+1) = A(k) x(k) + B(k) u(k)
%
% It also demonstrates sidLTVdiscTune for automatic regularization tuning.

%% LTI system recovery
% First, verify that sidLTVdisc correctly identifies a known LTI system
% where A and B are constant.

rng(100);
p = 2;  q = 1;  N = 50;  L = 10;
A_true = [0.9 0.1; -0.1 0.8];
B_true = [1; 0.5];
sigma = 0.02;

% Generate L trajectories
X = zeros(N+1, p, L);
U = randn(N, q, L);
for l = 1:L
    X(1, :, l) = randn(1, p);
    for k = 1:N
        X(k+1, :, l) = (A_true * X(k, :, l)' + B_true * U(k, :, l)' ...
            + sigma * randn(p, 1))';
    end
end

result_lti = sidLTVdisc(X, U, 'Lambda', 1e5);

% A(:,:,k) should be approximately constant
A_mean = mean(result_lti.A, 3);
fprintf('True A:\n');  disp(A_true);
fprintf('Mean recovered A:\n');  disp(A_mean);
fprintf('Recovery error: %.4f\n', norm(A_mean - A_true, 'fro'));

%% LTV system: time-varying pole
% The (1,1) entry of A ramps from 0.5 to 0.9 over time.

rng(200);
N = 80;  L = 15;
a_ramp = linspace(0.5, 0.9, N)';
A_tv = zeros(p, p, N);
for k = 1:N
    A_tv(:,:,k) = [a_ramp(k), 0.1; -0.1, 0.8];
end

X_tv = zeros(N+1, p, L);
U_tv = randn(N, q, L);
for l = 1:L
    X_tv(1, :, l) = randn(1, p);
    for k = 1:N
        X_tv(k+1, :, l) = (A_tv(:,:,k) * X_tv(k, :, l)' ...
            + B_true * U_tv(k, :, l)' + 0.05 * randn(p, 1))';
    end
end

result_tv = sidLTVdisc(X_tv, U_tv, 'Lambda', 1e4);

% Plot recovered A(1,1,k) vs true ramp
figure;
plot(1:N, squeeze(result_tv.A(1,1,:)), 'b', 'DisplayName', 'Recovered A_{11}(k)');
hold on;
plot(1:N, a_ramp, 'k--', 'LineWidth', 1.5, 'DisplayName', 'True A_{11}(k)');
xlabel('Time step k');
ylabel('A_{11}(k)');
title('LTV Identification: Time-Varying Pole Recovery');
legend('show');
grid on;
hold off;

%% Automatic lambda selection (L-curve)
% With 'Lambda', 'auto', sidLTVdisc uses the L-curve method to find the
% regularization that best balances data fidelity and smoothness.

result_auto = sidLTVdisc(X_tv, U_tv, 'Lambda', 'auto');
fprintf('\nAutomatic lambda: %.2e\n', result_auto.Lambda(1));

figure;
plot(1:N, squeeze(result_auto.A(1,1,:)), 'r', 'DisplayName', 'Auto \lambda');
hold on;
plot(1:N, squeeze(result_tv.A(1,1,:)), 'b', 'DisplayName', '\lambda = 10^4');
plot(1:N, a_ramp, 'k--', 'LineWidth', 1.5, 'DisplayName', 'True');
xlabel('Time step k');
ylabel('A_{11}(k)');
title('Manual vs Automatic Lambda');
legend('show');
grid on;
hold off;

%% Multi-trajectory benefit
% More trajectories provide more information and reduce estimation error.

rng(300);
L_few  = 3;
L_many = 20;

X_few  = X_tv(:, :, 1:L_few);
U_few  = U_tv(:, :, 1:L_few);

X_many = zeros(N+1, p, L_many);
U_many = randn(N, q, L_many);
for l = 1:L_many
    X_many(1, :, l) = randn(1, p);
    for k = 1:N
        X_many(k+1, :, l) = (A_tv(:,:,k) * X_many(k, :, l)' ...
            + B_true * U_many(k, :, l)' + 0.05 * randn(p, 1))';
    end
end

r_few  = sidLTVdisc(X_few,  U_few,  'Lambda', 1e4);
r_many = sidLTVdisc(X_many, U_many, 'Lambda', 1e4);

figure;
plot(1:N, squeeze(r_few.A(1,1,:)),  'r', 'DisplayName', sprintf('L = %d', L_few));
hold on;
plot(1:N, squeeze(r_many.A(1,1,:)), 'b', 'DisplayName', sprintf('L = %d', L_many));
plot(1:N, a_ramp, 'k--', 'LineWidth', 1.5, 'DisplayName', 'True');
xlabel('Time step k');
ylabel('A_{11}(k)');
title('Effect of Number of Trajectories');
legend('show');
grid on;
hold off;

%% Validation-based lambda tuning with sidLTVdiscTune
% Split trajectories into training and validation sets, then search over
% a grid of lambda values to minimize validation prediction error.

rng(400);
L_total = 20;  L_train = 14;  L_val = L_total - L_train;

X_all = zeros(N+1, p, L_total);
U_all = randn(N, q, L_total);
for l = 1:L_total
    X_all(1, :, l) = randn(1, p);
    for k = 1:N
        X_all(k+1, :, l) = (A_tv(:,:,k) * X_all(k, :, l)' ...
            + B_true * U_all(k, :, l)' + 0.05 * randn(p, 1))';
    end
end

X_train = X_all(:, :, 1:L_train);
U_train = U_all(:, :, 1:L_train);
X_val   = X_all(:, :, L_train+1:end);
U_val   = U_all(:, :, L_train+1:end);

grid_lam = logspace(0, 10, 30);
[bestResult, bestLambda, allLosses] = sidLTVdiscTune(X_train, U_train, ...
    X_val, U_val, 'LambdaGrid', grid_lam);

fprintf('\nValidation-tuned lambda: %.2e\n', bestLambda);

% Plot validation RMSE vs lambda
figure;
semilogx(grid_lam, allLosses, 'b.-');
hold on;
semilogx(bestLambda, min(allLosses), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
xlabel('\lambda');
ylabel('Validation RMSE');
title('Lambda Tuning: Validation Loss Curve');
grid on;

%% Preconditioning for numerical stability
% Block-diagonal preconditioning can improve conditioning of the solve.

result_pre = sidLTVdisc(X_tv, U_tv, 'Lambda', 1e4, 'Precondition', true);
fprintf('\nPreconditioned: %d\n', result_pre.Preconditioned);

%% Cost decomposition
% result.Cost = [total, data_fidelity, regularization]

fprintf('\nCost decomposition:\n');
fprintf('  Total:          %.4f\n', result_tv.Cost(1));
fprintf('  Data fidelity:  %.4f\n', result_tv.Cost(2));
fprintf('  Regularization: %.4f\n', result_tv.Cost(3));
fprintf('  Check: %.4e (should be ~0)\n', ...
    result_tv.Cost(1) - result_tv.Cost(2) - result_tv.Cost(3));
