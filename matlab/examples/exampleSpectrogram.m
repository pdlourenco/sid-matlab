%% exampleSpectrogram - Short-time FFT spectrogram on a chirp-driven SDOF
%
% Drives a physical SDOF plant (Plant D: m=1 kg, k=4e4 N/m, c=20 N.s/m,
% resonance at ~31.83 Hz) with a chirp force sweeping 20->60 Hz. The
% spectrogram of the response position shows the chirp ridge lighting
% up as it sweeps through the plant's resonance near t = 1.5 s.
%
% See spec/EXAMPLES.md section 3.8 for the binding specification.

runner__nCompleted = 0;

%% Chirp force driving the SDOF
% Build Plant D with util_msd and simulate under a linear chirp force.

m  = 1.0;
k  = 4e4;
c  = 20.0;
F  = 1.0;
Fs = 1000;
Ts = 1 / Fs;

[Ad, Bd] = util_msd(m, k, c, F, Ts);

N = 5000;
t = (0:N-1)' * Ts;
f0 = 20;
f1 = 60;
u_chirp = cos(2 * pi * (f0 + (f1 - f0) / (2 * t(end)) * t) .* t);

x = zeros(N + 1, 2);
for step = 1:N
    x(step + 1, :) = (Ad * x(step, :)' + Bd * u_chirp(step))';
end
y_chirp = x(2:end, 1);

result = sidSpectrogram(y_chirp, 'WindowLength', 256, 'SampleTime', Ts);

figure;
sidSpectrogramPlot(result);
title('SDOF response to chirp force: resonance lights up near ~32 Hz');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Chirp force driving the SDOF.\n', ...
    runner__nCompleted);

%% Window length trade-off
% Short window: good time resolution, poor frequency resolution. Long
% window: poor time resolution, good frequency resolution.

r_short = sidSpectrogram(y_chirp, 'WindowLength',  64, 'SampleTime', Ts);
r_long  = sidSpectrogram(y_chirp, 'WindowLength', 512, 'SampleTime', Ts);

figure;
subplot(1, 2, 1);
sidSpectrogramPlot(r_short, 'Axes', gca);
title('Short window (L = 64)');

subplot(1, 2, 2);
sidSpectrogramPlot(r_long, 'Axes', gca);
title('Long window (L = 512)');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Window length trade-off.\n', ...
    runner__nCompleted);

%% Window types
% Different windows trade main-lobe width against side-lobe suppression.

r_hann = sidSpectrogram(y_chirp, 'WindowLength', 256, 'Window', 'hann', ...
                         'SampleTime', Ts);
r_hamm = sidSpectrogram(y_chirp, 'WindowLength', 256, 'Window', 'hamming', ...
                         'SampleTime', Ts);
r_rect = sidSpectrogram(y_chirp, 'WindowLength', 256, 'Window', 'rect', ...
                         'SampleTime', Ts);

figure;
subplot(1, 3, 1);
sidSpectrogramPlot(r_hann, 'Axes', gca);
title('Hann');

subplot(1, 3, 2);
sidSpectrogramPlot(r_hamm, 'Axes', gca);
title('Hamming');

subplot(1, 3, 3);
sidSpectrogramPlot(r_rect, 'Axes', gca);
title('Rectangular');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Window types.\n', runner__nCompleted);

%% Multi-channel signal
% Channel 1: the chirp-force response from above. Channel 2: the same
% plant driven by a constant 50 Hz tone (steady-state response at
% 50 Hz).

u_fixed = cos(2 * pi * 50 * t);

x_fix = zeros(N + 1, 2);
for step = 1:N
    x_fix(step + 1, :) = (Ad * x_fix(step, :)' + Bd * u_fixed(step))';
end
y_fixed = x_fix(2:end, 1);

y_mc = [y_chirp, y_fixed];
result_mc = sidSpectrogram(y_mc, 'WindowLength', 256, 'SampleTime', Ts);

figure;
subplot(1, 2, 1);
sidSpectrogramPlot(result_mc, 'Channel', 1, 'Axes', gca);
title('Channel 1: chirp-force response (20 \rightarrow 60 Hz)');

subplot(1, 2, 2);
sidSpectrogramPlot(result_mc, 'Channel', 2, 'Axes', gca);
title('Channel 2: 50 Hz tone response');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Multi-channel signal.\n', runner__nCompleted);

%% Log frequency scale and NFFT zero-padding
% Zero-padding smooths the frequency axis by interpolating the STFT
% bins. Combined with a log frequency scale it emphasises the plant's
% low-frequency behaviour.

r_zp = sidSpectrogram(y_chirp, 'WindowLength', 128, 'NFFT', 1024, ...
                       'SampleTime', Ts);

figure;
sidSpectrogramPlot(r_zp, 'FrequencyScale', 'log');
title('Log frequency scale with zero-padding (NFFT = 1024)');

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Log frequency scale and NFFT zero-padding.\n', ...
    runner__nCompleted);

%% Accessing raw STFT data
fprintf('Spectrogram dimensions:\n');
fprintf('  Time points:    %d\n', length(result.Time));
fprintf('  Frequency bins: %d\n', length(result.Frequency));
fprintf('  Power shape:    %s\n', mat2str(size(result.Power)));
fprintf('  Complex shape:  %s\n', mat2str(size(result.Complex)));

runner__nCompleted = runner__nCompleted + 1;
fprintf('  Section %d completed: Accessing raw STFT data.\n', runner__nCompleted);

fprintf('exampleSpectrogram: %d/%d sections completed\n', ...
    runner__nCompleted, runner__nCompleted);
