# sid — Algorithm Specification

**Version:** 0.1.0-draft
**Date:** 2026-03-24
**Reference:** Ljung, L. *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999.

---

## 1. System Model

All frequency-domain estimation in this package assumes the general linear time-invariant model:

```
y(t) = G(q) u(t) + v(t)       t = 1, 2, ..., N
```

where:

- `y(t)` is the output signal, dimension `n_y × 1`
- `u(t)` is the input signal, dimension `n_u × 1`
- `G(q)` is the transfer function (transfer matrix for MIMO), dimension `n_y × n_u`
- `v(t)` is output disturbance noise, dimension `n_y × 1`, assumed independent of `u(t)`
- `q` is the forward shift operator: `q u(t) = u(t+1)`

The noise `v(t)` may optionally be modeled as filtered white noise:

```
v(t) = H(q) e(t)
```

where `e(t)` is white noise with covariance matrix `Λ`.

**Time series mode:** When no input is present (`n_u = 0`), the model reduces to `y(t) = v(t)` and only the output power spectrum is estimated.

**LTV extension:** The `sidFreqBTMap` function (§6) relaxes the time-invariance assumption by applying the Blackman-Tukey method to overlapping segments, producing a time-varying frequency response Ĝ(ω, t). Within each segment, local time-invariance is assumed.

---

## 2. `sidFreqBT` — Blackman-Tukey Spectral Analysis

### 2.1 Inputs

| Parameter | Symbol | Type | Default |
|-----------|--------|------|---------|
| Output data | `y` | `(N × n_y)` real matrix | required |
| Input data | `u` | `(N × n_u)` real matrix, or `[]` | `[]` (time series) |
| Window size | `M` | positive integer, `M ≥ 2` | `min(floor(N/10), 30)` |
| Frequencies | `ω` | `(n_f × 1)` vector, rad/sample | 128 points, see §2.2 |
| Sample time | `Ts` | positive scalar (seconds) | `1.0` |

All data must be real-valued and uniformly sampled. If `y` or `u` is a column vector, it is treated as a single channel.

### 2.2 Default Frequency Grid

When no frequency vector is specified, the default grid is 128 values **linearly** spaced in `(0, π]`:

```
ω_k = k × π / 128       k = 1, 2, ..., 128
```

in units of rad/sample. To convert to rad/s, divide by `Ts`:

```
ω_k (rad/s) = ω_k (rad/sample) / Ts
```

**Note on returned units:** The result struct stores frequencies in **rad/sample** internally. Plotting functions convert to rad/s using `Ts` when labeling axes.

**Rationale for linear spacing:** The FFT fast path (§2.5) produces linearly spaced frequency bins. Linear spacing is therefore the natural default that enables the FFT optimization. Users who want logarithmic spacing should pass an explicit frequency vector, which triggers the direct DFT path.

### 2.3 Covariance Estimation

Compute the biased sample cross-covariance between signals `x` and `z`, each of length `N`:

```
R̂_xz(τ) = (1/N) Σ_{t=1}^{N-|τ|} x(t+|τ|) z(t)       for τ ≥ 0
R̂_xz(τ) = conj(R̂_zx(-τ))                              for τ < 0
```

The biased estimator (dividing by `N` rather than `N-|τ|`) is used because:
1. It guarantees the resulting spectral estimate is non-negative.
2. It has lower mean-squared error than the unbiased estimator.

For the `sidFreqBT` algorithm, the following covariances are needed for lags `τ = 0, 1, ..., M`:

| Covariance | Signals | Dimensions | Used for |
|------------|---------|------------|----------|
| `R̂_y(τ)` | `y, y` | `n_y × n_y` | Output auto-spectrum |
| `R̂_u(τ)` | `u, u` | `n_u × n_u` | Input auto-spectrum |
| `R̂_yu(τ)` | `y, u` | `n_y × n_u` | Cross-spectrum |

**Time series mode** (`u = []`): Only `R̂_y(τ)` is computed.

### 2.4 Lag Window

The Hann (Hanning) window of size `M`:

```
W_M(τ) = 0.5 × (1 + cos(π τ / M))       for |τ| ≤ M
W_M(τ) = 0                                for |τ| > M
```

Properties:
- `W_M(0) = 1`
- `W_M(±M) = 0`
- Symmetric: `W_M(τ) = W_M(-τ)`
- Smooth taper to zero at the edges, reducing spectral leakage

The frequency resolution of the estimate is approximately `2π/M` rad/sample. Larger `M` gives finer resolution but higher variance.

### 2.5 Windowed Spectral Estimates

The spectral estimate at frequency `ω` is the Fourier transform of the windowed covariance:

```
Φ̂_xz(ω) = Σ_{τ=-M}^{M} R̂_xz(τ) × W_M(τ) × exp(-j ω τ)
```

This is computed for all three covariance pairs to produce `Φ̂_y(ω)`, `Φ̂_u(ω)`, and `Φ̂_yu(ω)`.

#### 2.5.1 FFT Fast Path

When using the default frequency grid (§2.2), the computation is done via FFT:

1. Construct the full windowed covariance sequence of length `2M+1`:
   ```
   c(τ) = R̂_xz(τ) × W_M(τ)       for τ = -M, ..., 0, ..., M
   ```

2. Arrange into FFT input order. For a length-`L` FFT where `L ≥ 2M+1`:
   ```
   s(k) = c(k)           for k = 0, 1, ..., M
   s(k) = 0              for k = M+1, ..., L-M-1     (zero-padding)
   s(k) = c(k - L)       for k = L-M, ..., L-1       (negative lags wrapped)
   ```
   In practice, `L = 256` (the smallest power of 2 ≥ 2×128+1, used for the default 128-frequency grid).

3. Compute `S = fft(s)`.

4. Extract the desired frequency bins: `Φ̂(ω_k) = S(k+1)` for `k = 1, ..., 128`
   (MATLAB 1-indexed: bin 1 is DC, bin `k+1` corresponds to frequency `k × 2π/L`).

**Scaling:** No additional scaling factor is applied. The FFT computes the sum directly.

#### 2.5.2 Direct DFT Path

When the user supplies a custom frequency vector `ω`, compute the sum explicitly:

```
Φ̂_xz(ω) = R̂_xz(0) × W_M(0) + 2 × Σ_{τ=1}^{M} Re[ R̂_xz(τ) × W_M(τ) × exp(-j ω τ) ]
```

Wait — this shortcut is only valid when `R̂_xz(τ)` is the auto-covariance of a real signal (which is real and symmetric). For the cross-covariance `R̂_yu(τ)`, the full complex sum must be used:

```
Φ̂_yu(ω) = Σ_{τ=-M}^{M} R̂_yu(τ) × W_M(τ) × exp(-j ω τ)
```

where `R̂_yu(-τ) = R̂_uy(τ)' = conj(R̂_uy(τ))` for the scalar case.

**Implementation:** For each frequency `ω_k`, compute:

```
Φ̂_xz(ω_k) = W_M(0) × R̂_xz(0) + Σ_{τ=1}^{M} W_M(τ) × [ R̂_xz(τ) × exp(-j ω_k τ)
                                                             + conj(R̂_xz(τ)) × exp(+j ω_k τ) ]
```

which for real signals and auto-covariances simplifies to:

```
Φ̂_xx(ω_k) = W_M(0) × R̂_xx(0) + 2 × Σ_{τ=1}^{M} W_M(τ) × R̂_xx(τ) × cos(ω_k τ)
```

This form is real-valued and non-negative, as expected for a power spectrum.

### 2.6 Frequency Response Estimate

**SISO case:**

```
Ĝ(ω) = Φ̂_yu(ω) / Φ̂_u(ω)
```

**MIMO case** (`n_u > 1`):

```
Ĝ(ω) = Φ̂_yu(ω) × Φ̂_u(ω)^{-1}
```

where `Φ̂_yu(ω)` is `n_y × n_u` and `Φ̂_u(ω)` is `n_u × n_u`. The matrix inverse is computed independently at each frequency.

**Regularization:** If `Φ̂_u(ω)` is singular or nearly singular at some frequency `ω_k`:
- SISO: if `|Φ̂_u(ω_k)| < ε × max(|Φ̂_u|)` where `ε = 1e-10`, set `Ĝ(ω_k) = NaN + j×NaN`.
- MIMO: if `cond(Φ̂_u(ω_k)) > 1/ε`, set the corresponding row of `Ĝ(ω_k)` to `NaN`.
- Issue a warning when this occurs.

### 2.7 Noise Spectrum Estimate

**SISO case:**

```
Φ̂_v(ω) = Φ̂_y(ω) - |Φ̂_yu(ω)|² / Φ̂_u(ω)
```

**MIMO case:**

```
Φ̂_v(ω) = Φ̂_y(ω) - Φ̂_yu(ω) × Φ̂_u(ω)^{-1} × Φ̂_yu(ω)'
```

where `'` denotes conjugate transpose.

**Non-negativity:** Due to estimation errors, `Φ̂_v(ω)` may become slightly negative at some frequencies. Clamp to zero:

```
Φ̂_v(ω) = max(Φ̂_v(ω), 0)      (SISO)
```

For MIMO, ensure the matrix is positive semi-definite by zeroing any negative eigenvalues.

**Time series mode:** No noise spectrum is computed separately. The output spectrum `Φ̂_y(ω)` is returned in the `NoiseSpectrum` field.

### 2.8 Normalization

The spectral estimates use the following normalization:

```
Φ̂(ω) = Σ_{τ=-M}^{M} R̂(τ) W_M(τ) exp(-jωτ)
```

This matches the System Identification Toolbox convention. It does **not** include:
- A factor of `Ts` (the Signal Processing Toolbox convention includes `Ts`)
- A factor of `1/(2π)`

To convert to the Signal Processing Toolbox convention, multiply by `Ts`:

```
Φ̂_SPT(ω) = Ts × Φ̂_SID(ω)
```

---

## 3. Uncertainty Estimation

### 3.1 Window Norm

Define the squared window norm:

```
C_W = Σ_{τ=-M}^{M} W_M(τ)²
```

For the Hann window, this evaluates to:

```
C_W = 1 + 2 × Σ_{τ=1}^{M} [0.5 × (1 + cos(πτ/M))]²
```

which can be computed in closed form as `C_W = (3/4)×(2M) + 1/2 = (3M + 1)/2`, but the implementation should compute it numerically from the actual window values to avoid any discrepancy.

### 3.2 Coherence

The squared coherence between input and output:

```
γ̂²(ω) = |Φ̂_yu(ω)|² / (Φ̂_y(ω) × Φ̂_u(ω))
```

This is real-valued and satisfies `0 ≤ γ̂²(ω) ≤ 1`. Values near 1 indicate the output is well explained by the input at that frequency; values near 0 indicate noise dominates.

### 3.3 Variance of the Frequency Response

The asymptotic variance of the frequency response estimate (Ljung 1999, p. 184):

```
Var{Ĝ(ω)} ≈ (C_W / N) × |Ĝ(ω)|² × (1 - γ̂²(ω)) / γ̂²(ω)
```

The standard deviation returned in the result struct is:

```
σ_G(ω) = sqrt(Var{Ĝ(ω)})
```

**Regularization:** If `γ̂²(ω_k) < ε` (where `ε = 1e-10`), set `σ_G(ω_k) = Inf`. This corresponds to frequencies where the input has negligible power and the estimate is unreliable.

**Note:** This formula gives the variance of the complex-valued `Ĝ`. The standard deviation `σ_G` applies equally to real and imaginary parts. Confidence intervals for magnitude are constructed as:

```
|Ĝ(ω)| ± p × σ_G(ω)
```

where `p` is the number of standard deviations (default: 3 for ≈99.7% coverage under Gaussian assumptions).

### 3.4 Variance of the Noise Spectrum

The asymptotic variance of the spectral estimate (Ljung 1999, p. 188):

```
Var{Φ̂_v(ω)} ≈ (2 × C_W / N) × Φ̂_v(ω)²
```

Standard deviation:

```
σ_Φv(ω) = sqrt(Var{Φ̂_v(ω)})
```

### 3.5 Variance of the Output Spectrum (Time Series Mode)

When no input is present:

```
Var{Φ̂_y(ω)} ≈ (2 × C_W / N) × Φ̂_y(ω)²
```

This is the standard asymptotic result for windowed spectral estimates.

---

## 4. `sidFreqETFE` — Empirical Transfer Function Estimate

### 4.1 Algorithm

The ETFE is the ratio of the output and input discrete Fourier transforms:

```
Ĝ_ETFE(ω_k) = Y(ω_k) / U(ω_k)
```

where:

```
Y(ω_k) = Σ_{t=1}^{N} y(t) exp(-j ω_k t)
U(ω_k) = Σ_{t=1}^{N} u(t) exp(-j ω_k t)
```

This is equivalent to `sidFreqBT` with window size `M = N` (rectangular window). It provides the maximum frequency resolution but has high variance.

### 4.2 Optional Smoothing

A smoothing parameter `S` (positive odd integer) may be specified. When given, the raw ETFE is convolved with a length-`S` rectangular (boxcar) frequency-domain window:

```
Ĝ_smooth(ω_k) = (1/S) × Σ_{j=-(S-1)/2}^{(S-1)/2} Ĝ_ETFE(ω_{k+j})
```

with appropriate handling at the boundaries.

### 4.3 Noise Spectrum

For the ETFE, the noise spectrum estimate is the periodogram of the residuals:

```
Φ̂_v(ω_k) = (1/N) × |Y(ω_k) - Ĝ(ω_k) × U(ω_k)|²
```

### 4.4 Time Series Mode

When no input is present, the ETFE reduces to the **periodogram**:

```
Φ̂_y(ω_k) = (1/N) × |Y(ω_k)|²
```

---

## 5. `sidFreqBTFDR` — Frequency-Dependent Resolution

### 5.1 Concept

`sidFreqBTFDR` is identical to `sidFreqBT` except that the window size `M` varies with frequency, allowing different resolution at different frequencies. The user specifies a **resolution** parameter `R(ω)` (in rad/sample) instead of a window size.

### 5.2 Resolution to Window Size Mapping

At each frequency `ω_k`, the local window size is:

```
M_k = ceil(2π / R_k)
```

where `R_k = R(ω_k)` is the desired resolution at that frequency.

If `R` is a scalar, it applies uniformly. If `R` is a vector of the same length as the frequency grid, each entry specifies the local resolution.

### 5.3 Algorithm

For each frequency `ω_k`:

1. Determine `M_k` from the resolution.
2. Compute the Hann window `W_{M_k}(τ)` of size `M_k`.
3. Compute the windowed spectral estimates `Φ̂_y(ω_k)`, `Φ̂_u(ω_k)`, `Φ̂_yu(ω_k)` using the direct DFT formula with window size `M_k`.
4. Form `Ĝ(ω_k)` and `Φ̂_v(ω_k)` as in §2.6 and §2.7.

**Note:** The FFT fast path cannot be used here because the window size varies across frequencies. All computations use the direct DFT.

### 5.4 Default Resolution

If no resolution is specified:

```
R = 2π / min(floor(N/10), 30)
```

This matches the default behavior of `sidFreqBT`.

---

## 6. `sidFreqBTMap` — Time-Varying Frequency Response Map

### 6.1 Concept

`sidFreqBTMap` applies `sidFreqBT` to overlapping segments of data, producing a **time-varying frequency response estimate** Ĝ(ω, t). This reveals how the system's transfer function, noise spectrum, and coherence evolve over time.

For an LTI system, the map is constant along the time axis — this serves as a diagnostic check. For an LTV (linear time-varying) system, the map shows modes appearing, disappearing, shifting in frequency, or changing in gain.

This extends the `spectrogram` concept from single-signal time-frequency analysis to **input-output system identification**:

| Tool | Input | Output | Shows |
|------|-------|--------|-------|
| `spectrogram` / `sidSpectrogram` | One signal | \|X(ω,t)\|² | How signal frequency content changes |
| `sidFreqBTMap` | Input + output pair | Ĝ(ω,t), Φ̂_v(ω,t), γ̂²(ω,t) | How the *system itself* changes |
| `sidFreqBTMap` | One signal (time series) | Φ̂_y(ω,t) | How signal spectrum changes (≈ spectrogram) |

When used together, `sidSpectrogram` on `u` and `y` alongside `sidFreqBTMap` on the pair `(y, u)` provides a complete diagnostic picture: the input's spectral content, the output's spectral content, and the system connecting them — all on aligned time axes.

### 6.2 Inputs

| Parameter | Symbol | Type | Default |
|-----------|--------|------|---------|
| Output data | `y` | `(N × n_y)` real matrix | required |
| Input data | `u` | `(N × n_u)` real matrix, or `[]` | `[]` (time series) |
| Segment length | `L` | positive integer | `min(floor(N/4), 256)` |
| Overlap | `P` | integer, `0 ≤ P < L` | `floor(L/2)` (50% overlap) |
| Window size | `M` | positive integer, `M ≥ 2` | `min(floor(L/10), 30)` |
| Frequencies | `ω` | `(n_f × 1)` vector, rad/sample | 128 points (default grid) |
| Sample time | `Ts` | positive scalar (seconds) | `1.0` |

### 6.3 Algorithm

1. Divide the data into `K` overlapping segments, each of length `L` samples, with overlap `P`:
   ```
   Segment k: samples (k-1)(L-P)+1  through  (k-1)(L-P)+L
   for k = 1, 2, ..., K
   where K = floor((N - L) / (L - P)) + 1
   ```

2. For each segment `k`, extract `y_k = y(start:end, :)` and `u_k = u(start:end, :)`.

3. Run `sidFreqBT(y_k, u_k, M, freqs, Ts)` on each segment.

4. Collect the per-segment results into time-frequency arrays.

### 6.4 Time Vector

The center time of each segment defines the time axis:

```
t_k = ((k-1)(L-P) + L/2) × Ts       for k = 1, ..., K
```

in units of seconds.

### 6.5 Output Struct

`sidFreqBTMap` returns a struct with fields:

| Field | Type | Description |
|-------|------|-------------|
| `Time` | `(K × 1)` real | Center time of each segment (seconds) |
| `Frequency` | `(n_f × 1)` real | Frequency vector (rad/sample) |
| `FrequencyHz` | `(n_f × 1)` real | Frequency vector (Hz) |
| `Response` | `(n_f × K)` complex | Time-varying frequency response Ĝ(ω, t) |
| `ResponseStd` | `(n_f × K)` real | Standard deviation of Ĝ per segment |
| `NoiseSpectrum` | `(n_f × K)` real | Time-varying noise spectrum Φ̂_v(ω, t) |
| `NoiseSpectrumStd` | `(n_f × K)` real | Standard deviation per segment |
| `Coherence` | `(n_f × K)` real | Time-varying squared coherence γ̂²(ω, t) |
| `SampleTime` | scalar | Sample time Ts |
| `SegmentLength` | scalar | Segment length L |
| `Overlap` | scalar | Overlap P |
| `WindowSize` | scalar | BT window size M |
| `Method` | char | `'sidFreqBTMap'` |

**Dimensions shown are for SISO.** For MIMO, `Response` becomes `(n_f × K × n_y × n_u)`, etc.

### 6.6 Visualization: `sidMapPlot`

The natural visualization is a **color map** (like a spectrogram):

- **x-axis:** Time (seconds)
- **y-axis:** Frequency (rad/s or Hz, log scale)
- **Color:** Magnitude of Ĝ(ω, t) in dB, or Φ̂_v(ω, t) in dB, or γ̂²(ω, t)

The function `sidMapPlot` provides selectable plot types via a `'PlotType'` option:

| PlotType | Color represents | Use case |
|----------|-----------------|----------|
| `'magnitude'` (default) | `20 log10(\|Ĝ(ω,t)\|)` | Track gain changes |
| `'phase'` | `angle(Ĝ(ω,t))` in degrees | Track phase drift |
| `'noise'` | `10 log10(Φ̂_v(ω,t))` | Track disturbance evolution |
| `'coherence'` | `γ̂²(ω,t)` on [0, 1] | Identify when LTI assumption breaks down |
| `'spectrum'` | `10 log10(Φ̂_y(ω,t))` | Time series mode (equivalent to spectrogram) |

### 6.7 Relationship to `spectrogram`

In time series mode (`u = []`), `sidFreqBTMap` computes Φ̂_y(ω, t) over sliding windows. This is functionally similar to a spectrogram, but computed via the Blackman-Tukey (windowed correlogram) method rather than the short-time FFT (Welch/periodogram) method. The two approaches differ in their bias-variance characteristics and in how windowing is applied (lag-domain vs. time-domain), but produce qualitatively similar results.

The key differentiator is that with input-output data, `sidFreqBTMap` produces quantities that `spectrogram` cannot: the time-varying transfer function, noise spectrum, and coherence.

### 6.8 Design Considerations

**Segment length vs. window size:** The BT window size `M` controls the frequency resolution within each segment; the segment length `L` determines how much data is used per estimate. The requirement `L > 2M` must hold. For good estimates, `L >> M` is preferred — a typical choice is `L = 10M` or more.

**Computational cost:** `K` calls to `sidFreqBT`, each operating on `L` samples. For default parameters this is fast, but for large `N` with small step size `L-P`, the number of segments can be large. The implementation should pre-compute covariances per segment efficiently rather than recomputing from scratch.

**Edge effects:** The first and last segments may produce less reliable estimates if the system is non-stationary near the boundaries. No special handling is applied — the uncertainty estimates from each segment naturally reflect the reduced confidence.

---

## 7. `sidSpectrogram` — Short-Time Spectral Analysis

### 7.1 Purpose

`sidSpectrogram` computes the short-time Fourier transform (STFT) spectrogram of one or more signals. It replicates the core functionality of the Signal Processing Toolbox `spectrogram` function, with two additional roles in the `sid` workflow:

1. **Diagnostic companion to `sidFreqBTMap`.** Plotting the spectrograms of `y` and `u` alongside the time-varying transfer function map lets the user distinguish genuine system changes from input-driven effects. If a spectral feature appears in both the `y` spectrogram and the Ĝ(ω,t) map but *not* in the `u` spectrogram, it's likely a real system change. If it appears in `u` too, it's the input driving the output.

2. **Standalone time-frequency analysis** for users who don't have the Signal Processing Toolbox.

### 7.2 Inputs

| Parameter | Symbol | Type | Default |
|-----------|--------|------|---------|
| Signal | `x` | `(N × n_ch)` real matrix | required |
| Window length | `L` | positive integer | `256` |
| Overlap | `P` | integer, `0 ≤ P < L` | `floor(L/2)` |
| NFFT | `nfft` | positive integer | `max(256, 2^nextpow2(L))` |
| Window function | `win` | `'hann'`, `'hamming'`, `'rect'`, or `(L × 1)` vector | `'hann'` |
| Sample time | `Ts` | positive scalar (seconds) | `1.0` |

**Note on window terminology:** The window here is a **time-domain** tapering window applied to each data segment before FFT — this is distinct from the **lag-domain** Hann window used in `sidFreqBT`. The spectrogram window reduces spectral leakage; the BT lag window controls frequency resolution of the correlogram.

### 7.3 Algorithm

The standard short-time Fourier transform:

1. Divide the signal `x` into `K` overlapping segments of length `L`, with overlap `P`:
   ```
   x_k(n) = x((k-1)(L-P) + n) × w(n)       n = 1, ..., L
   ```
   where `w(n)` is the time-domain window and `K = floor((N - L) / (L - P)) + 1`.

2. Compute the FFT of each windowed segment:
   ```
   X_k(m) = Σ_{n=1}^{L} x_k(n) × exp(-j 2π (m-1) n / nfft)       m = 1, ..., nfft
   ```

3. Compute the one-sided power spectral density for each segment:
   ```
   P_k(m) = (1 / (Fs × S₁)) × |X_k(m)|²
   ```
   where `S₁ = Σ w(n)²` is the window power, and `Fs = 1/Ts`. For one-sided spectra, the positive-frequency bins (excluding DC and Nyquist) are doubled.

4. The spectrogram is the matrix `P(m, k)` for `m = 1, ..., nfft/2+1` and `k = 1, ..., K`.

### 7.4 Output Struct

| Field | Type | Description |
|-------|------|-------------|
| `Time` | `(K × 1)` real | Center time of each segment (seconds) |
| `Frequency` | `(n_bins × 1)` real | Frequency vector (Hz) |
| `FrequencyRad` | `(n_bins × 1)` real | Frequency vector (rad/s) |
| `Power` | `(n_bins × K × n_ch)` real | Power spectral density per segment |
| `PowerDB` | `(n_bins × K × n_ch)` real | `10 × log10(Power)` |
| `Complex` | `(n_bins × K × n_ch)` complex | Complex STFT coefficients (before squaring) |
| `SampleTime` | scalar | Sample time Ts |
| `WindowLength` | scalar | Segment length L |
| `Overlap` | scalar | Overlap P |
| `NFFT` | scalar | FFT length |
| `Method` | char | `'sidSpectrogram'` |

where `n_bins = floor(nfft/2) + 1` (one-sided spectrum).

### 7.5 Visualization

`sidSpectrogram` can be plotted using `sidMapPlot` with `'PlotType', 'spectrum'`, or with a dedicated call:

```matlab
result = sidSpectrogram(y, 'WindowLength', 256, 'Overlap', 128);
sidSpectrogramPlot(result);
```

`sidSpectrogramPlot` produces a standard spectrogram color map:

- **x-axis:** Time (seconds)
- **y-axis:** Frequency (Hz), linear or log scale
- **Color:** Power in dB

### 7.6 Relationship to `sidFreqBTMap`

The two functions share segmentation conventions (segment length, overlap, time vector computation) so their time axes align when called with the same parameters. A typical diagnostic workflow:

```matlab
% Same segmentation parameters for alignment
L = 256; P = 128; Ts = 0.001;

% Spectrograms of raw signals
specY = sidSpectrogram(y, 'WindowLength', L, 'Overlap', P, 'SampleTime', Ts);
specU = sidSpectrogram(u, 'WindowLength', L, 'Overlap', P, 'SampleTime', Ts);

% Time-varying transfer function
mapG = sidFreqBTMap(y, u, 'SegmentLength', L, 'Overlap', P, 'SampleTime', Ts);

% Compare side-by-side
figure;
subplot(3,1,1); sidSpectrogramPlot(specU); title('Input u');
subplot(3,1,2); sidSpectrogramPlot(specY); title('Output y');
subplot(3,1,3); sidMapPlot(mapG, 'PlotType', 'magnitude'); title('G(w,t)');
```

This layout immediately reveals whether spectral features in the output are input-driven or system-driven.

### 7.7 Compatibility with MathWorks `spectrogram`

The MathWorks `spectrogram` function uses the calling convention `spectrogram(x, window, noverlap, nfft, fs)`. `sidSpectrogram` supports a compatible positional syntax:

```matlab
% MathWorks style:
[S, F, T, P] = spectrogram(x, hann(256), 128, 512, 1000);

% sid equivalent:
result = sidSpectrogram(x, 'WindowLength', 256, 'Overlap', 128, ...
                           'NFFT', 512, 'SampleTime', 0.001);
% result.Complex ≈ S, result.Frequency ≈ F, result.Time ≈ T, result.Power ≈ P
```

The normalization follows the PSD convention (power per unit frequency), matching the MathWorks default when `spectrogram` is called with the `'psd'` option.

---

## 8. Output Struct

All `sidFreq*` functions return a struct with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `Frequency` | `(n_f × 1)` real | Frequency vector in rad/sample |
| `FrequencyHz` | `(n_f × 1)` real | Frequency vector in Hz: `ω / (2π Ts)` |
| `Response` | `(n_f × n_y × n_u)` complex | Frequency response `Ĝ(ω)` |
| `ResponseStd` | `(n_f × n_y × n_u)` real | Standard deviation of `Ĝ` |
| `NoiseSpectrum` | `(n_f × n_y × n_y)` real | Noise spectrum `Φ̂_v(ω)` or `Φ̂_y(ω)` |
| `NoiseSpectrumStd` | `(n_f × n_y × n_y)` real | Standard deviation of noise spectrum |
| `Coherence` | `(n_f × 1)` real | Squared coherence `γ̂²(ω)` (SISO only, `[]` for MIMO) |
| `SampleTime` | scalar | Sample time `Ts` in seconds |
| `WindowSize` | scalar or vector | Window size `M` (scalar for BT, vector for BTFDR) |
| `DataLength` | scalar | Number of samples `N` |
| `Method` | char | `'sidFreqBT'`, `'sidFreqBTFDR'`, or `'sidFreqETFE'` |

**Dimension conventions:**
- SISO: `Response` is `(n_f × 1)`, `NoiseSpectrum` is `(n_f × 1)`.
- MIMO: Dimensions are `(n_f × n_y × n_u)` for `Response` and `(n_f × n_y × n_y)` for `NoiseSpectrum`.

**Time series mode:** `Response` and `ResponseStd` are empty (`[]`). `Coherence` is empty. `NoiseSpectrum` contains `Φ̂_y(ω)`.

---

## 9. Edge Cases and Validation

### 9.1 Input Validation

| Condition | Action |
|-----------|--------|
| `N < 2 × M` | Reduce `M` to `floor(N/2)` and issue warning |
| `M < 2` | Error: window size must be at least 2 |
| `size(y,1) ~= size(u,1)` | Error: input and output must have same number of samples |
| `N < 10` | Warning: very short data, estimates will be unreliable |
| `y` or `u` contains NaN or Inf | Error: data must be finite |
| `y` or `u` is not real | Error: complex data not supported in v1.0 |
| Any frequency `ω_k ≤ 0` or `ω_k > π` | Error: frequencies must be in (0, π] rad/sample |
| `Ts ≤ 0` | Error: sample time must be positive |

### 9.2 Numerical Edge Cases

| Condition | Action |
|-----------|--------|
| `Φ̂_u(ω_k) ≈ 0` | Set `Ĝ(ω_k) = NaN`, `σ_G(ω_k) = Inf`, issue warning |
| `Φ̂_v(ω_k) < 0` | Clamp to 0 |
| `γ̂²(ω_k) > 1` (numerical error) | Clamp to 1 |
| `γ̂²(ω_k) < 0` (numerical error) | Clamp to 0 |

### 9.3 Degenerate Inputs

| Condition | Action |
|-----------|--------|
| `u` is constant (zero variance) | Same as `Φ̂_u ≈ 0` at all frequencies; `Ĝ = NaN` everywhere, with warning |
| `y` is constant | Valid; `Φ̂_y ≈ 0` at all frequencies |
| `u = y` (perfect coherence) | Valid; `γ̂² ≈ 1`, `Φ̂_v ≈ 0`, very small `σ_G` |

---

## 10. Plotting

### 10.1 `sidBodePlot`

Produces a two-panel figure:
- **Top panel:** Magnitude `20 × log10(|Ĝ(ω)|)` in dB vs. frequency
- **Bottom panel:** Phase `angle(Ĝ(ω)) × 180/π` in degrees vs. frequency

Both panels use logarithmic frequency axis (rad/s by default, Hz if requested).

Confidence bands are shown as a shaded region at `±p` standard deviations (default `p = 3`):
- Magnitude band: `20 × log10(|Ĝ| ± p × σ_G)` — note this is applied to the linear magnitude, then converted to dB.
- Phase band: `±p × σ_G / |Ĝ| × 180/π` — small-angle approximation for phase uncertainty.

### 10.2 `sidSpectrumPlot`

Single panel: `10 × log10(Φ̂_v(ω))` in dB vs. frequency (log axis).

Confidence band: `10 × log10(Φ̂_v ± p × σ_Φv)` — applied in linear scale, converted to dB.

### 10.3 Options

Both plotting functions accept name-value options:

| Option | Default | Description |
|--------|---------|-------------|
| `'Confidence'` | `3` | Number of standard deviations for shaded band |
| `'FrequencyUnit'` | `'rad/s'` | `'rad/s'` or `'Hz'` |
| `'ShowConfidence'` | `true` | Whether to show the confidence band |
| `'Color'` | MATLAB default | Line color |
| `'LineWidth'` | `1.5` | Line width |
| `'Axes'` | `[]` | Axes handle (creates new figure if empty) |

---

## 11. References

1. Ljung, L. *System Identification: Theory for the User*, 2nd ed. Prentice Hall, 1999.
   - §2.3: Spectral analysis fundamentals
   - §6.3–6.4: Non-parametric frequency-domain methods
   - Table 6.1: Default window sizes
   - p. 184: Asymptotic variance of frequency response estimate
   - p. 188: Asymptotic variance of spectral estimate

2. Blackman, R.B. and Tukey, J.W. *The Measurement of Power Spectra*. Dover, 1959.

3. Kay, S.M. *Modern Spectral Estimation: Theory and Application*. Prentice Hall, 1988.

4. Stoica, P. and Moses, R.L. *Spectral Analysis of Signals*. Prentice Hall, 2005.