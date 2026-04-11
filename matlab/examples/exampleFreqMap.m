%% exampleFreqMap - Time-varying frequency response maps on SMD plants
%
% Four physical scenarios demonstrating sidFreqMap:
%   1. LTI baseline: 2-mass SMD (should look stationary in time).
%   2. Continuous LTV: k1(t) ramps 200 -> 20 N/m over the record.
%   3. Discrete LTV: k1 step change 200 -> 40 N/m at t = T/2.
%   4. Duffing hardening SDOF: apparent resonance rises with amplitude
%      under ramped-amplitude excitation, purely from the nonlinearity.
%
% Sections 2-3 use util_msd_ltv; section 4 uses util_msd_nl.
%
% See spec/EXAMPLES.md section 3.7 for the binding specification.

runner__nCompleted = 0;

%% 1. LTI baseline: constant 2-mass chain
% Plant B with F = [1;0]. The magnitude map should look constant
% along the time axis.

rng(10);

m  = [1; 1];
k  = [100; 80];
c  = [2; 2];
F  = [1; 0];
Ts = 0.01;
N  = 4000;

[Ad, Bd] = util_msd(m, k, c, F, Ts);

u = randn(N, 1);
x = zeros(N + 1, 4);
for step = 1:N
    x(step + 1, :) = (Ad * x(step, :)' + Bd * u(step))';
end
y = x(2:end, 1) + 5e-4 * randn(N, 1);

result_lti = sidFreqMap(y, u, 'SegmentLength', 512, 'Overlap', 384, ...
                         'SampleTime', Ts);

figure;
sidMapPlot(result_lti, 'PlotType', 'magnitude');
title('LTI 2-mass chain: magnitude should be constant along time');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: LTI baseline.\n', runner__nCompleted);

%% 2. Continuous LTV: ramping first-spring stiffness
% k1(t) ramps from 200 to 20 N/m over the record. We build the per-
% step discrete dynamics stack with util_msd_ltv and simulate the
% LTV recursion.

rng(20);

k_tv = zeros(2, N);
k_tv(1, :) = linspace(200, 20, N);
k_tv(2, :) = 80;
m_tv = repmat(m, 1, N);
c_tv = repmat(c, 1, N);

[Ad_tv, Bd_tv] = util_msd_ltv(m_tv, k_tv, c_tv, F, Ts);

u_tv = randn(N, 1);
x_tv = zeros(N + 1, 4);
for step = 1:N
    x_tv(step + 1, :) = (Ad_tv(:, :, step) * x_tv(step, :)' ...
        + Bd_tv(:, :, step) * u_tv(step))';
end
y_tv = x_tv(2:end, 1) + 5e-4 * randn(N, 1);

result_ramp = sidFreqMap(y_tv, u_tv, 'SegmentLength', 256, 'Overlap', 192, ...
                          'SampleTime', Ts);

figure;
sidMapPlot(result_ramp, 'PlotType', 'magnitude');
title('Continuous LTV: k_1 ramps 200 \rightarrow 20 (mode drifts 7 \rightarrow 3 rad/s)');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Continuous LTV.\n', runner__nCompleted);

%% 3. Discrete LTV: step change in stiffness
% Same plant but k1 snaps from 200 to 40 at t = T/2. Caricature of a
% sudden structural change (cable cut, latch release).

rng(30);

k_step = zeros(2, N);
k_step(1, 1:N/2)       = 200;
k_step(1, N/2 + 1:end) = 40;
k_step(2, :) = 80;

[Ad_step, Bd_step] = util_msd_ltv(m_tv, k_step, c_tv, F, Ts);

u_step = randn(N, 1);
x_step = zeros(N + 1, 4);
for step = 1:N
    x_step(step + 1, :) = (Ad_step(:, :, step) * x_step(step, :)' ...
        + Bd_step(:, :, step) * u_step(step))';
end
y_step = x_step(2:end, 1) + 5e-4 * randn(N, 1);

result_step = sidFreqMap(y_step, u_step, 'SegmentLength', 256, ...
                          'Overlap', 192, 'SampleTime', Ts);

figure;
sidMapPlot(result_step, 'PlotType', 'magnitude');
title('Discrete LTV: step change k_1: 200 \rightarrow 40 at t = T/2');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Discrete LTV.\n', runner__nCompleted);

%% Coherence map
% Coherence shows how the signal-to-noise ratio evolves over time.
% Reuse the ramping-stiffness run.

figure;
sidMapPlot(result_ramp, 'PlotType', 'coherence');
title('Coherence map: ramping stiffness');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Coherence map.\n', runner__nCompleted);

%% BT vs Welch algorithm
% sidFreqMap supports 'bt' (correlogram-based) and 'welch' (sub-
% segment FFT averaging).

result_bt    = sidFreqMap(y_tv, u_tv, 'SegmentLength', 256, ...
                          'Algorithm', 'bt',    'SampleTime', Ts);
result_welch = sidFreqMap(y_tv, u_tv, 'SegmentLength', 256, ...
                          'Algorithm', 'welch', 'SampleTime', Ts);

figure;
subplot(1, 2, 1);
sidMapPlot(result_bt, 'PlotType', 'magnitude', 'Axes', gca);
title('Blackman-Tukey');

subplot(1, 2, 2);
sidMapPlot(result_welch, 'PlotType', 'magnitude', 'Axes', gca);
title('Welch');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: BT vs Welch algorithm.\n', runner__nCompleted);

%% Segment length and overlap tuning
% Shorter segments = better time resolution. Longer segments = better
% frequency resolution.

result_short = sidFreqMap(y_tv, u_tv, 'SegmentLength', 128, ...
                           'Overlap', 96,  'SampleTime', Ts);
result_long  = sidFreqMap(y_tv, u_tv, 'SegmentLength', 512, ...
                           'Overlap', 384, 'SampleTime', Ts);

figure;
subplot(1, 2, 1);
sidMapPlot(result_short, 'PlotType', 'magnitude', 'Axes', gca);
title('Short segments (L = 128): good time resolution');

subplot(1, 2, 2);
sidMapPlot(result_long, 'PlotType', 'magnitude', 'Axes', gca);
title('Long segments (L = 512): good frequency resolution');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Segment length and overlap tuning.\n', ...
    runner__nCompleted);

%% Time-series mode: evolving output spectrum
% Fresh simulation of the ramping-stiffness plant; hand only the
% output to sidFreqMap.

rng(40);
u_nu = randn(N, 1);
x_nu = zeros(N + 1, 4);
for step = 1:N
    x_nu(step + 1, :) = (Ad_tv(:, :, step) * x_nu(step, :)' ...
        + Bd_tv(:, :, step) * u_nu(step))';
end
y_nu = x_nu(2:end, 1);

result_ts = sidFreqMap(y_nu, [], 'SegmentLength', 256, 'Overlap', 192, ...
                        'SampleTime', Ts);

figure;
sidMapPlot(result_ts, 'PlotType', 'spectrum');
title('Time-series: output spectrum drifts as k_1 softens');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Time-series mode.\n', runner__nCompleted);

%% 4. Duffing hardening oscillator
% Nonlinear SDOF with cubic stiffness: m*x'' + c*x' + k_lin*x +
% k_cub*x^3 = u(t). For a hardening spring (k_cub > 0), the
% effective resonance frequency rises with amplitude:
%   omega_eff(x) = sqrt( (k_lin + 3*k_cub*x^2) / m )
% We drive the oscillator with a ramped-amplitude white force
% (amplitude grows 0.5 -> 10 over the record). As typical
% displacement grows, the apparent resonance should drift upward.

rng(50);

m_nl    = 1.0;
k_lin   = 100.0;
k_cubic = 1e5;
c_nl    = 2.0;
F_nl    = 1.0;

amp  = linspace(0.5, 10.0, N)';
u_nl = amp .* randn(N, 1);

x_nl = util_msd_nl(m_nl, k_lin, k_cubic, c_nl, F_nl, Ts, u_nl, ...
                   [], 4);
y_nl = x_nl(2:end, 1);

fprintf('Typical early amplitude: %.4f m\n', std(y_nl(1:N/4)));
fprintf('Typical late amplitude:  %.4f m\n', std(y_nl(end - N/4 + 1:end)));
fprintf('Linearized omega_n (small amplitude):  %.2f rad/s\n', ...
    sqrt(k_lin / m_nl));
fprintf('Effective omega_n at 0.05 m amplitude: %.2f rad/s\n', ...
    sqrt((k_lin + 3 * k_cubic * 0.05^2) / m_nl));

result_nl = sidFreqMap(y_nl, u_nl, 'SegmentLength', 256, 'Overlap', 192, ...
                        'SampleTime', Ts);

figure;
sidMapPlot(result_nl, 'PlotType', 'magnitude');
title('Duffing hardening: apparent resonance rises with amplitude');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Duffing hardening oscillator.\n', ...
    runner__nCompleted);

fprintf('exampleFreqMap: %d/%d sections completed\n', ...
    runner__nCompleted, runner__nCompleted);
