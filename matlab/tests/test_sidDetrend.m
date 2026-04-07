% test_sidDetrend.m - Test polynomial detrending function

fprintf('Running test_sidDetrend...\n');
runner__nPassed = 0;

%% Test 1: Linear trend removal
N = 500;
t = (0:N-1)';
slope = 2.5; intercept = 10;
trend_true = slope * t + intercept;
noise = 0.1 * randn(N, 1);
x = trend_true + noise;

[x_dt, trend_est] = sidDetrend(x);

% The detrended signal should have near-zero mean and no linear trend
assert(abs(mean(x_dt)) < 0.5, 'Detrended mean should be near zero');
% The removed trend should match the true trend closely
assert(max(abs(trend_est - trend_true)) < 1.0, ...
    'Removed trend should approximate true linear trend');
% Reconstruction: x = x_dt + trend
assert(max(abs(x - (x_dt + trend_est))) < 1e-12, ...
    'x = x_detrended + trend must hold exactly');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 1 passed: linear trend removal.\n');

%% Test 2: Mean removal (Order=0)
rng(5001);
N = 1000;
x = 42 + randn(N, 1);  % mean = 42
x_dm = sidDetrend(x, 'Order', 0);

assert(abs(mean(x_dm)) < 1e-10, 'Mean should be removed to machine precision');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 2 passed: mean removal (Order=0).\n');

%% Test 3: Quadratic trend
N = 500;
t = (0:N-1)';
trend_true = 0.001 * t.^2 - 0.5 * t + 100;
x = trend_true + 0.01 * randn(N, 1);

[x_dt, trend_est] = sidDetrend(x, 'Order', 2);
assert(max(abs(trend_est - trend_true)) < 0.5, ...
    'Quadratic trend should be well approximated');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 3 passed: quadratic trend removal.\n');

%% Test 4: Multi-channel
rng(5002);
N = 300;
t = (0:N-1)';
x = [3*t + 10 + randn(N,1), -2*t + 50 + randn(N,1), 0.5*t.^2 + randn(N,1)];

[x_dt, trend] = sidDetrend(x);
assert(isequal(size(x_dt), [N, 3]), 'Output should be N x 3');
assert(max(abs(x(:) - (x_dt(:) + trend(:)))) < 1e-12, ...
    'Reconstruction must hold for all channels');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 4 passed: multi-channel detrending.\n');

%% Test 5: Segment-wise detrending
rng(5003);
N = 600;
t = (0:N-1)';
% Piecewise linear trend: slope changes at t=300
trend_true = zeros(N, 1);
trend_true(1:300) = 2 * (0:299)';
trend_true(301:600) = trend_true(300) - 1.5 * (0:299)';
x = trend_true + 0.5 * randn(N, 1);

x_dt = sidDetrend(x, 'SegmentLength', 300);
% Each segment should have its local trend removed
assert(abs(mean(x_dt(1:300))) < 2, 'First segment mean should be small');
assert(abs(mean(x_dt(301:600))) < 2, 'Second segment mean should be small');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 5 passed: segment-wise detrending.\n');

%% Test 6: Multi-trajectory (3D)
rng(5004);
N = 200; L = 4;
t = (0:N-1)';
x3 = zeros(N, 1, L);
for l = 1:L
    x3(:, 1, l) = (l * 0.5) * t + 10*l + randn(N, 1);
end

[x3_dt, trend3] = sidDetrend(x3);
assert(isequal(size(x3_dt), [N, 1, L]), 'Output should preserve 3D shape');
assert(max(abs(x3(:) - (x3_dt(:) + trend3(:)))) < 1e-12, ...
    'Reconstruction must hold for 3D input');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 6 passed: multi-trajectory (3D) detrending.\n');

%% Test 7: Already zero-mean data
rng(5005);
N = 200;
x = randn(N, 1);
x = x - mean(x);  % exact zero mean

x_dt = sidDetrend(x, 'Order', 0);
assert(max(abs(x - x_dt)) < 1e-10, 'Zero-mean data should be unchanged');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 7 passed: already zero-mean data unchanged.\n');

%% Test 8: Trend output reconstruction
rng(5006);
N = 100;
x = 5 * (0:N-1)' + randn(N, 1);
[x_dt, trend] = sidDetrend(x);
assert(max(abs(x - (x_dt + trend))) < 1e-12, 'x = x_dt + trend exactly');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 8 passed: trend output reconstruction.\n');

%% Test 9: High polynomial degree (Order=5)
rng(5009);
N = 200;
t = linspace(0, 1, N)';
% Generate a 5th-degree polynomial trend
trend_true = 3*t.^5 - 2*t.^4 + t.^3 - 0.5*t.^2 + 0.1*t + 7;
x = trend_true + 0.001 * randn(N, 1);

[x_dt, trend_est] = sidDetrend(x, 'Order', 5);
assert(max(abs(trend_est - trend_true)) < 0.1, ...
    'Order-5 trend should be well recovered');
assert(max(abs(x - (x_dt + trend_est))) < 1e-12, ...
    'Reconstruction must hold');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 9 passed: high polynomial degree (Order=5).\n');

%% Test 10: Order >= N clamped gracefully
rng(5010);
N = 5;
x = randn(N, 1);

% Order=10 > N=5, should be clamped to N-1=4 internally
[x_dt, trend] = sidDetrend(x, 'Order', 10);
assert(isequal(size(x_dt), [N, 1]), 'Output size should be N x 1');
% With degree clamped to N-1, the polynomial fits the data perfectly
% so detrended should be near zero
assert(max(abs(x_dt)) < 1e-8, ...
    'Order >= N should fit data perfectly, detrended near zero');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 10 passed: Order >= N clamped gracefully.\n');

%% Test 11: Segment length not dividing N evenly
rng(5011);
N = 100;
t = (0:N-1)';
x = 2 * t + randn(N, 1);

[x_dt, trend] = sidDetrend(x, 'SegmentLength', 30);
% N=100, SegLen=30: segments [1-30],[31-60],[61-90],[91-100]
assert(isequal(size(x_dt), [N, 1]), 'Output size correct');
assert(max(abs(x - (x_dt + trend))) < 1e-12, 'Reconstruction holds');
% Last segment (10 samples) should still be detrended
assert(abs(mean(x_dt(91:100))) < 5, ...
    'Last short segment should be detrended');
runner__nPassed = runner__nPassed + 1;
fprintf('  Test 11 passed: segment length not dividing N.\n');

fprintf('test_sidDetrend: %d/%d passed\n', runner__nPassed, runner__nPassed);
