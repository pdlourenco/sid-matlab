%% exampleMIMO - MIMO frequency response on a 2-mass chain
%
% Demonstrates MIMO system identification with sidFreqBT on Plant B
% (a 2-mass SMD chain). First a 2-output / 1-input case (force at
% mass 1, measure both positions), then the full 2x2 MIMO case with
% force actuators on both masses.
%
% See spec/EXAMPLES.md section 3.6 for the binding specification.

runner__nCompleted = 0;

%% 2-output, 1-input system
% Force at mass 1 only, measure positions of both masses. The result
% has Response shape (nf, 2, 1). Static gains should match the first
% column of inv(K) (the compliance matrix).

rng(10);

m  = [1; 1];
k  = [100; 80];
c  = [2; 2];
F  = [1; 0];      % force at mass 1 only
Ts = 0.01;
N  = 4000;

[Ad, Bd] = util_msd(m, k, c, F, Ts);
C_out = [1 0 0 0; 0 1 0 0];

u = randn(N, 1);
x = zeros(N + 1, 4);
for step = 1:N
    x(step + 1, :) = (Ad * x(step, :)' + Bd * u(step))';
end
y = x(2:end, 1:2) + 2e-4 * randn(N, 2);

w_grid = linspace(0.005, pi, 512)';
result = sidFreqBT(y, u, 'WindowSize', 200, 'Frequencies', w_grid, ...
                   'SampleTime', Ts);

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: 2-output, 1-input system.\n', ...
    runner__nCompleted);

%% Inspect MIMO result dimensions
fprintf('Response shape:       %s  (nf, ny, nu)\n', ...
    mat2str(size(result.Response)));
fprintf('Noise spectrum shape: %s  (nf, ny, ny)\n', ...
    mat2str(size(result.NoiseSpectrum)));
fprintf('Coherence is empty:   %d\n', isempty(result.Coherence));

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Inspect MIMO result dimensions.\n', ...
    runner__nCompleted);

%% Plot individual output channels
% Two-panel Bode magnitude: G1 (input -> x1) and G2 (input -> x2)
% with the dashed true TF overlay on each.

w = result.Frequency;
nf = length(w);
G1_true = zeros(nf, 1);
G2_true = zeros(nf, 1);
I4 = eye(4);
for i = 1:nf
    Mi = (exp(1j * w(i)) * I4 - Ad) \ Bd;
    G1_true(i) = C_out(1, :) * Mi;
    G2_true(i) = C_out(2, :) * Mi;
end

figure;
subplot(2, 1, 1);
semilogx(w, 20*log10(abs(result.Response(:, 1, 1))), 'b', ...
    'DisplayName', 'Estimated');
hold on;
semilogx(w, 20*log10(abs(G1_true)), 'k--', 'DisplayName', 'True');
hold off;
ylabel('Magnitude (dB)');
title('G_1: force at mass 1 \rightarrow position x_1');
legend;
grid on;

subplot(2, 1, 2);
semilogx(w, 20*log10(abs(result.Response(:, 2, 1))), 'r', ...
    'DisplayName', 'Estimated');
hold on;
semilogx(w, 20*log10(abs(G2_true)), 'k--', 'DisplayName', 'True');
hold off;
ylabel('Magnitude (dB)');
xlabel('Frequency (rad/sample)');
title('G_2: force at mass 1 \rightarrow position x_2');
legend;
grid on;

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Plot individual output channels.\n', ...
    runner__nCompleted);

%% Noise spectral matrix
% The diagonal entries of the (nf, 2, 2) noise spectrum.

diag11 = real(result.NoiseSpectrum(:, 1, 1));
diag22 = real(result.NoiseSpectrum(:, 2, 2));

figure;
semilogx(w, 10*log10(diag11), 'b', 'DisplayName', '\Phi_{v,11}');
hold on;
semilogx(w, 10*log10(diag22), 'r', 'DisplayName', '\Phi_{v,22}');
hold off;
xlabel('Frequency (rad/sample)');
ylabel('Noise spectrum (dB)');
title('Diagonal elements of the noise spectral matrix');
legend;
grid on;

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Noise spectral matrix.\n', ...
    runner__nCompleted);

%% 2-output, 2-input system
% Independent white forces at both masses, measure both positions.
% util_msd handles multi-input plants directly.

rng(20);

F_22 = [1 0; 0 1];
[Ad_22, Bd_22] = util_msd(m, k, c, F_22, Ts);

N2 = 4000;
u2in = randn(N2, 2);
x2 = zeros(N2 + 1, 4);
for step = 1:N2
    x2(step + 1, :) = (Ad_22 * x2(step, :)' + Bd_22 * u2in(step, :)')';
end
y2out = x2(2:end, 1:2) + 2e-4 * randn(N2, 2);

result_22 = sidFreqBT(y2out, u2in, 'WindowSize', 200, ...
                      'Frequencies', w_grid, 'SampleTime', Ts);

fprintf('2x2 MIMO response shape: %s\n', mat2str(size(result_22.Response)));

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: 2-output, 2-input system.\n', ...
    runner__nCompleted);

%% Plot the full 2x2 transfer matrix
% Four subplots: direct compliances G11, G22 on the diagonal,
% cross-couplings G12, G21 on the off-diagonal.

titles = {'G_{11}: u_1 -> y_1', 'G_{12}: u_2 -> y_1'; ...
          'G_{21}: u_1 -> y_2', 'G_{22}: u_2 -> y_2'};

figure;
for iy = 1:2
    for iu = 1:2
        subplot(2, 2, (iy - 1) * 2 + iu);
        semilogx(w, 20*log10(abs(result_22.Response(:, iy, iu))), 'b');
        ylabel('Magnitude (dB)');
        if iy == 2
            xlabel('Frequency (rad/sample)');
        end
        title(titles{iy, iu});
        grid on;
    end
end

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Plot the full 2x2 transfer matrix.\n', ...
    runner__nCompleted);

%% MIMO uncertainty
% In v1.0 the MATLAB sidFreqBT computes per-pair asymptotic standard
% deviations for MIMO. The Python port returns NaN for MIMO (no
% asymptotic formula implemented there). We report what this
% implementation actually produced.

n_nan = sum(isnan(result_22.ResponseStd(:)));
n_tot = numel(result_22.ResponseStd);
fprintf('MIMO ResponseStd: %d of %d entries are NaN\n', n_nan, n_tot);

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: MIMO uncertainty.\n', runner__nCompleted);

fprintf('exampleMIMO: %d/%d sections completed\n', ...
    runner__nCompleted, runner__nCompleted);
