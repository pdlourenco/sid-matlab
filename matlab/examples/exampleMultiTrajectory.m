%% exampleMultiTrajectory - Multi-trajectory ensemble averaging on SMD plants
%
% Four sub-sections demonstrate the 1/sqrt(L) variance reduction
% achievable by ensemble averaging:
%
%   1. LTI ensemble averaging on Plant B (2-mass chain) with sidFreqBT.
%   2. LTV time-varying map on Plant B with a step change in k1, using
%      sidFreqMap.
%   3. Spectrogram averaging on Plant D (SDOF) driven by a chirp force
%      buried in heavy measurement noise.
%   4. COSMIC + sidFreqMap consistency on the 1-DoF ltv_disc LTV plant.
%
% See spec/EXAMPLES.md section 3.10 for the binding specification.

runner__nCompleted = 0;

%% 1. LTI ensemble averaging: tighter confidence bands
% L = 10 trajectories of Plant B. Compare sidFreqBT confidence bands
% between a single trajectory and the full ensemble. The ensemble's
% ResponseStd should be about 1/sqrt(L) times the single-trajectory
% value.

rng(5001);

m  = [1; 1];  k = [100; 80];  c = [2; 2];  F = [1; 0];
Ts = 0.01;
N = 2000;  L = 10;
[Ad, Bd] = util_msd(m, k, c, F, Ts);

y_all = zeros(N, 1, L);
u_all = zeros(N, 1, L);
for l = 1:L
    u_all(:, 1, l) = randn(N, 1);
    xs = zeros(N + 1, 4);
    for step = 1:N
        xs(step + 1, :) = (Ad * xs(step, :)' + Bd * u_all(step, 1, l))';
    end
    y_all(:, 1, l) = xs(2:end, 1) + 5e-4 * randn(N, 1);
end

% Single trajectory
r1 = sidFreqBT(y_all(:, :, 1), u_all(:, :, 1), ...
               'WindowSize', 80, 'SampleTime', Ts);

% Multi-trajectory ensemble
rL = sidFreqBT(y_all, u_all, 'WindowSize', 80, 'SampleTime', Ts);

max1 = max(r1.ResponseStd(~isnan(r1.ResponseStd) & ~isinf(r1.ResponseStd)));
maxL = max(rL.ResponseStd(~isnan(rL.ResponseStd) & ~isinf(rL.ResponseStd)));
fprintf('Single trajectory: max ResponseStd = %.3e\n', max1);
fprintf('%d trajectories:   max ResponseStd = %.3e\n', L, maxL);
fprintf('Ratio: %.2f (expected ~%.2f = 1/sqrt(%d))\n', ...
    maxL / max1, 1 / sqrt(L), L);

figure;
ax_m1 = subplot(2, 2, 1);
ax_p1 = subplot(2, 2, 3);
sidBodePlot(r1, 'Confidence', 3, 'Axes', [ax_m1, ax_p1]);
title(ax_m1, 'Single trajectory');

ax_mL = subplot(2, 2, 2);
ax_pL = subplot(2, 2, 4);
sidBodePlot(rL, 'Confidence', 3, 'Axes', [ax_mL, ax_pL]);
title(ax_mL, sprintf('Ensemble of %d trajectories', L));

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: LTI ensemble averaging.\n', ...
    runner__nCompleted);

%% 2. LTV time-varying map: sharper transition detection
% Plant B with a step change in k1 at t = N/2. More trajectories
% sharpen the transition in the freq_map.

rng(5002);
N_tv = 4000;  L_tv = 5;

k_step = zeros(2, N_tv);
k_step(1, 1:N_tv/2)       = 200;
k_step(1, N_tv/2 + 1:end) = 50;
k_step(2, :) = 80;
m_tv = repmat(m, 1, N_tv);
c_tv = repmat(c, 1, N_tv);

[Ad_tv, Bd_tv] = util_msd_ltv(m_tv, k_step, c_tv, F, Ts);

y_ltv = zeros(N_tv, 1, L_tv);
u_ltv = zeros(N_tv, 1, L_tv);
for l = 1:L_tv
    u_ltv(:, 1, l) = randn(N_tv, 1);
    xs = zeros(N_tv + 1, 4);
    for step = 1:N_tv
        xs(step + 1, :) = (Ad_tv(:, :, step) * xs(step, :)' ...
            + Bd_tv(:, 1, step) * u_ltv(step, 1, l))';
    end
    y_ltv(:, 1, l) = xs(2:end, 1) + 5e-4 * randn(N_tv, 1);
end

r1_map = sidFreqMap(y_ltv(:, :, 1), u_ltv(:, :, 1), ...
    'SegmentLength', 256, 'SampleTime', Ts);
rL_map = sidFreqMap(y_ltv, u_ltv, 'SegmentLength', 256, 'SampleTime', Ts);

figure;
subplot(1, 2, 1);
sidMapPlot(r1_map, 'PlotType', 'magnitude', 'Axes', gca);
title('Single trajectory');

subplot(1, 2, 2);
sidMapPlot(rL_map, 'PlotType', 'magnitude', 'Axes', gca);
title(sprintf('Ensemble of %d trajectories', L_tv));

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: LTV time-varying map.\n', runner__nCompleted);

%% 3. Spectrogram averaging: chirp in noise
% Plant D (SDOF, omega_n = 200 rad/s) driven by a chirp force, with
% heavy measurement noise added to the response. Single trajectory:
% chirp barely visible. L = 8 trajectories averaged: chirp track
% emerges clearly.

rng(5003);
N_sp = 2000;  L_sp = 8;  Fs = 1000;
Ts_sp = 1 / Fs;
t_sp = (0:N_sp-1)' * Ts_sp;

m_sp = 1.0;  k_sp = 4e4;  c_sp = 20.0;  F_sp = 1.0;
[Ad_sp, Bd_sp] = util_msd(m_sp, k_sp, c_sp, F_sp, Ts_sp);

f0 = 20;  f1 = 60;
u_chirp = cos(2 * pi * (f0 + (f1 - f0) / (2 * t_sp(end)) * t_sp) .* t_sp);

x_all = zeros(N_sp, 1, L_sp);
for l = 1:L_sp
    xs = zeros(N_sp + 1, 2);
    for step = 1:N_sp
        xs(step + 1, :) = (Ad_sp * xs(step, :)' + Bd_sp * u_chirp(step))';
    end
    x_all(:, 1, l) = xs(2:end, 1) + 1e-4 * randn(N_sp, 1);
end

r1_spec = sidSpectrogram(x_all(:, :, 1), 'WindowLength', 128, ...
                          'SampleTime', Ts_sp);
rL_spec = sidSpectrogram(x_all, 'WindowLength', 128, ...
                          'SampleTime', Ts_sp);

figure;
subplot(1, 2, 1);
sidSpectrogramPlot(r1_spec, 'Axes', gca);
title('Single trajectory');

subplot(1, 2, 2);
sidSpectrogramPlot(rL_spec, 'Axes', gca);
title(sprintf('Ensemble of %d trajectories', L_sp));

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Spectrogram averaging.\n', ...
    runner__nCompleted);

%% 4. COSMIC + sidFreqMap consistency
% Reuse the 1-DoF LTV plant (Plant A with ramping stiffness) and run
% both COSMIC and sidFreqMap on the same dataset.

rng(5004);
p = 2;  q = 1;
N_co = 80;  L_co = 10;
k_co = linspace(200.0, 50.0, N_co);
[Ad_co, Bd_co] = util_msd_ltv(1.0, k_co, 2.0, 1.0, Ts);

X = zeros(N_co + 1, p, L_co);
U = randn(N_co, q, L_co);
for l = 1:L_co
    X(1, :, l) = randn(1, p);
    for step = 1:N_co
        X(step + 1, :, l) = (Ad_co(:, :, step) * X(step, :, l)' ...
            + Bd_co(:, :, step) * U(step, :, l)' + 0.01 * randn(p, 1))';
    end
end

ltv = sidLTVdisc(X, U, 'Lambda', 'auto', 'Uncertainty', true);
fprintf('COSMIC identified A(:,:,1):\n');
disp(ltv.A(:, :, 1));
fprintf('COSMIC identified A(:,:,end):\n');
disp(ltv.A(:, :, end));

% Ensemble sidFreqMap on position channel
y_freq = X(1:N_co, 1, :);
u_freq = U;
fmap = sidFreqMap(y_freq, u_freq, 'SegmentLength', min(N_co, 30), ...
                   'SampleTime', Ts);
fprintf('sidFreqMap ensemble: %d trajectories, %d time segments.\n', ...
    fmap.NumTrajectories, length(fmap.Time));
fprintf('Both COSMIC and sidFreqMap use the same %d-trajectory dataset.\n', L_co);

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: COSMIC + sidFreqMap consistency.\n', ...
    runner__nCompleted);

fprintf('exampleMultiTrajectory: %d/%d sections completed\n', ...
    runner__nCompleted, runner__nCompleted);
