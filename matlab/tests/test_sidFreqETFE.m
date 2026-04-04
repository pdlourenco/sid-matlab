%% test_sidFreqETFE - Unit tests for Empirical Transfer Function Estimate
%
% Tests sidFreqETFE for result structure, smoothing, time-series mode,
% and correctness against known systems.

fprintf('Running test_sidFreqETFE...\n');

%% Test 1: Result struct has all required fields
rng(42);
N = 500;
u = randn(N, 1);
y = filter([1], [1 -0.8], u) + 0.1 * randn(N, 1);
result = sidFreqETFE(y, u);

requiredFields = {'Frequency', 'FrequencyHz', 'Response', 'ResponseStd', ...
    'NoiseSpectrum', 'NoiseSpectrumStd', 'Coherence', 'SampleTime', ...
    'WindowSize', 'DataLength', 'Method'};
for i = 1:length(requiredFields)
    assert(isfield(result, requiredFields{i}), 'Missing field: %s', requiredFields{i});
end

%% Test 2: Method and metadata
assert(strcmp(result.Method, 'sidFreqETFE'), 'Method should be sidFreqETFE');
assert(result.DataLength == N, 'DataLength should be N');
assert(result.WindowSize == N, 'WindowSize should be N for ETFE');

%% Test 3: SISO dimensions
nf = length(result.Frequency);
assert(isequal(size(result.Response), [nf, 1]), 'Response should be (nf x 1)');
assert(isempty(result.Coherence), 'ETFE Coherence should be empty');

%% Test 4: ResponseStd is NaN (no asymptotic formula for ETFE)
assert(all(isnan(result.ResponseStd)), 'ETFE ResponseStd should be NaN');

%% Test 5: Noise spectrum is non-negative
assert(all(result.NoiseSpectrum >= 0), 'Noise spectrum should be non-negative');

%% Test 6: Time series mode (periodogram)
rng(10);
y = randn(200, 1);
result_ts = sidFreqETFE(y, []);
assert(isempty(result_ts.Response), 'Time series: Response should be empty');
assert(all(result_ts.NoiseSpectrum >= 0), 'Periodogram should be non-negative');

%% Test 7: Smoothing parameter
rng(42);
N = 500;
u = randn(N, 1);
y = filter([1], [1 -0.8], u);
result_s1 = sidFreqETFE(y, u, 'Smoothing', 1);
result_s11 = sidFreqETFE(y, u, 'Smoothing', 11);
% Smoothed version should have less variation
var_s1 = var(abs(result_s1.Response));
var_s11 = var(abs(result_s11.Response));
assert(var_s11 < var_s1, 'Smoothing should reduce variance of response');

%% Test 8: Error on even smoothing parameter
try
    sidFreqETFE(y, u, 'Smoothing', 4);
    error('Should have thrown sid:badSmoothing');
catch e
    assert(strcmp(e.identifier, 'sid:badSmoothing'), 'Expected sid:badSmoothing error');
end

%% Test 9: Error on non-integer smoothing
try
    sidFreqETFE(y, u, 'Smoothing', 3.5);
    error('Should have thrown sid:badSmoothing');
catch e
    assert(strcmp(e.identifier, 'sid:badSmoothing'), 'Expected sid:badSmoothing error');
end

%% Test 10: Custom frequencies
w = linspace(0.1, pi, 50)';
result_cust = sidFreqETFE(y, u, 'Frequencies', w);
assert(length(result_cust.Frequency) == 50, 'Should have 50 custom frequencies');

%% Test 11: Known system - pure gain
% y = 2*u (no noise, no dynamics)
rng(5);
N = 1024;
u = randn(N, 1);
y = 2 * u;
result = sidFreqETFE(y, u);
% G should be approximately 2 at all frequencies
G_mag = abs(result.Response);
assert(max(abs(G_mag - 2)) < 0.01, 'ETFE of pure gain y=2u should give |G|~2');

%% Test 12: Known system - pure delay
% y(t) = u(t-1), so G(z) = z^{-1}, |G| = 1, phase = -w
rng(5);
N = 1024;
u = randn(N, 1);
y = [0; u(1:end-1)];
result = sidFreqETFE(y, u);
G_mag = abs(result.Response);
G_phase = angle(result.Response);
expected_phase = -result.Frequency;
% Magnitude should be ~1
assert(median(abs(G_mag - 1)) < 0.05, 'ETFE of pure delay: |G| should be ~1');
% Phase should be ~-w (allow some tolerance)
phase_err = abs(G_phase - expected_phase);
% Wrap phase errors
phase_err = min(phase_err, 2*pi - phase_err);
assert(median(phase_err) < 0.1, 'ETFE of pure delay: phase should be ~-w');

%% Test 13: MIMO mode
rng(7);
N = 500;
u = randn(N, 1);
y = [filter([1], [1 -0.5], u), filter([0.3], [1 -0.7], u)];
result_mimo = sidFreqETFE(y, u);
nf = length(result_mimo.Frequency);
assert(size(result_mimo.Response, 1) == nf && size(result_mimo.Response, 2) == 2, ...
    'MIMO Response should be (nf x 2 x 1)');

%% Test 14: Custom sample time
result = sidFreqETFE(y, u, 'SampleTime', 0.001);
assert(result.SampleTime == 0.001, 'SampleTime should be 0.001');

%% Test 15: Multi-trajectory — NumTrajectories and ensemble averaging
rng(15);
N15 = 2000; L15 = 8;
u15 = randn(N15, 1, L15);
y15 = zeros(N15, 1, L15);
for l = 1:L15
    y15(:, :, l) = filter(1, [1 -0.5], u15(:, :, l)) ...
        + 0.3 * randn(N15, 1);
end

res_mt = sidFreqETFE(y15, u15);
assert(res_mt.NumTrajectories == L15, ...
    'NumTrajectories should be %d, got %d', L15, res_mt.NumTrajectories);

res_st = sidFreqETFE(y15(:, :, 1), u15(:, :, 1));
assert(res_st.NumTrajectories == 1, 'Single traj should have NumTrajectories=1');

% Ensemble-averaged magnitude should be closer to truth
w = res_mt.Frequency;
G_true = 1 ./ (1 - 0.5 * exp(-1j * w));
err_mt = median(abs(abs(res_mt.Response) - abs(G_true)));
err_st = median(abs(abs(res_st.Response) - abs(G_true)));
assert(err_mt < err_st * 1.5, ...
    'Multi-traj error %.4f should improve on single %.4f', ...
    err_mt, err_st);

fprintf('  Test 15 passed: multi-trajectory ETFE (L=%d).\n', L15);

fprintf('  test_sidFreqETFE: ALL PASSED\n');
