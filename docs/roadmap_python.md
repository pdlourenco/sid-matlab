# sid — Python Port Roadmap

This document tracks all phases required to achieve feature parity with the
MATLAB/Octave v1.0 implementation. The authoritative algorithm specification
is `spec/SPEC.md`. The MATLAB code in `matlab/sid/` is the ground truth for
numerical behaviour.

---

## Naming Convention

Python functions follow the same `sid` + `Domain` + `Method` pattern, translated
to snake_case. Result field names are also snake_case.

```
sid.freq_bt(y, u, window_size=30)      # sidFreqBT(y, u, 'WindowSize', 30)
result.response                         # result.Response
result.noise_spectrum                   # result.NoiseSpectrum
```

### Function Catalog (v1.0 scope)

| MATLAB function | Python function | Description | Status |
|----------------|-----------------|-------------|--------|
| **`sidFreqBT`** | `sid.freq_bt` | Frequency response via Blackman-Tukey | ⬜ |
| **`sidFreqBTFDR`** | `sid.freq_btfdr` | Blackman-Tukey, frequency-dependent resolution | ⬜ |
| **`sidFreqETFE`** | `sid.freq_etfe` | Empirical transfer function estimate | ⬜ |
| **`sidFreqMap`** | `sid.freq_map` | Time-varying frequency response map (BT or Welch) | ⬜ |
| **`sidSpectrogram`** | `sid.spectrogram` | Short-time FFT spectrogram | ⬜ |
| **`sidLTVdisc`** | `sid.ltv_disc` | Discrete LTV state-space identification (COSMIC) | ⬜ |
| **`sidLTVdiscTune`** | `sid.ltv_disc_tune` | Lambda tuning (validation-based and frequency-response) | ⬜ |
| **`sidLTVdiscFrozen`** | `sid.ltv_disc_frozen` | Frozen transfer function G(ω,k) from A(k), B(k) | ⬜ |
| **`sidLTVdiscIO`** | `sid.ltv_disc_io` | Partial-observation LTV identification (Output-COSMIC) | ⬜ |
| **`sidLTVStateEst`** | `sid.ltv_state_est` | Batch LTV state estimation (RTS smoother) | ⬜ |
| **`sidLTIfreqIO`** | `sid.lti_freq_io` | LTI realization from I/O frequency response (Ho-Kalman) | ⬜ |
| **`sidModelOrder`** | `sid.model_order` | Model order estimation (Hankel SVD) | ⬜ |
| **`sidDetrend`** | `sid.detrend` | Polynomial detrending (preprocessing) | ⬜ |
| **`sidResidual`** | `sid.residual` | Residual analysis (whiteness + independence tests) | ⬜ |
| **`sidCompare`** | `sid.compare` | Model output comparison with fit metric | ⬜ |

### Plotting Functions

| MATLAB function | Python function | Description | Status |
|----------------|-----------------|-------------|--------|
| **`sidBodePlot`** | `sid.bode_plot` | Bode diagram with confidence bands | ⬜ |
| **`sidSpectrumPlot`** | `sid.spectrum_plot` | Power spectrum with confidence bands | ⬜ |
| **`sidMapPlot`** | `sid.map_plot` | Time-frequency color map | ⬜ |
| **`sidSpectrogramPlot`** | `sid.spectrogram_plot` | Spectrogram color map | ⬜ |

### Private Helper Functions

| MATLAB function | Python module | Description |
|----------------|---------------|-------------|
| `sidValidateData` | `_internal/validate_data.py` | Data validation and orientation |
| `sidHannWin` | `_internal/hann_win.py` | Hann lag window |
| `sidCov` | `_internal/cov.py` | Biased cross-covariance estimation |
| `sidDFT` | `_internal/dft.py` | DFT at arbitrary frequencies |
| `sidIsDefaultFreqs` | `_internal/is_default_freqs.py` | Default frequency grid detection |
| `sidWindowedDFT` | `_internal/windowed_dft.py` | Windowed Fourier transform (FFT + direct paths) |
| `sidUncertainty` | `_internal/uncertainty.py` | Asymptotic variance formulas |
| `sidParseOptions` | *(not needed — Python uses kwargs)* | — |
| `sidTestMSD` | `_internal/test_msd.py` | Mass-spring-damper test system (ZOH discretization) |
| `sidLTVbuildDataMatrices` | `_internal/ltv_build_data_matrices.py` | COSMIC data matrix construction |
| `sidLTVbuildDataMatricesVarLen` | `_internal/ltv_build_data_matrices.py` | Variable-length trajectory variant |
| `sidLTVbuildBlockTerms` | `_internal/ltv_build_block_terms.py` | COSMIC block Hessian terms |
| `sidLTVcosmicSolve` | `_internal/ltv_cosmic_solve.py` | COSMIC forward-backward solver |
| `sidLTVevaluateCost` | `_internal/ltv_evaluate_cost.py` | COSMIC cost function evaluation |
| `sidLTVblkTriSolve` | `_internal/ltv_blk_tri_solve.py` | Generic block-tridiagonal solver |
| `sidLTVuncertaintyBackwardPass` | `_internal/ltv_uncertainty_backward_pass.py` | Posterior covariance recursion |
| `sidEstimateNoiseCov` | `_internal/estimate_noise_cov.py` | Noise covariance from COSMIC residuals |
| `sidExtractStd` | `_internal/extract_std.py` | Standard deviations of A(k), B(k) |
| `sidFreqDomainSim` | `_internal/freq_domain_sim.py` | Frequency-domain simulation via IFFT |

---

## Package Structure

```
python/
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── sid/
│   ├── __init__.py                  # Public API re-exports
│   ├── _results.py                  # Frozen dataclasses for result types
│   ├── _exceptions.py               # SidError exception class
│   ├── freq_bt.py
│   ├── freq_etfe.py
│   ├── freq_btfdr.py
│   ├── freq_map.py
│   ├── spectrogram.py
│   ├── detrend.py
│   ├── ltv_disc.py
│   ├── ltv_disc_io.py
│   ├── ltv_disc_frozen.py
│   ├── ltv_disc_tune.py
│   ├── lti_freq_io.py
│   ├── ltv_state_est.py
│   ├── residual.py
│   ├── compare.py
│   ├── model_order.py
│   ├── bode_plot.py
│   ├── spectrum_plot.py
│   ├── map_plot.py
│   ├── spectrogram_plot.py
│   └── _internal/
│       ├── __init__.py
│       ├── validate_data.py
│       ├── hann_win.py
│       ├── cov.py
│       ├── dft.py
│       ├── is_default_freqs.py
│       ├── windowed_dft.py
│       ├── uncertainty.py
│       ├── test_msd.py
│       ├── ltv_build_data_matrices.py
│       ├── ltv_build_block_terms.py
│       ├── ltv_cosmic_solve.py
│       ├── ltv_evaluate_cost.py
│       ├── ltv_blk_tri_solve.py
│       ├── ltv_uncertainty_backward_pass.py
│       ├── estimate_noise_cov.py
│       ├── extract_std.py
│       └── freq_domain_sim.py
└── tests/
    ├── conftest.py
    ├── test_hann_win.py
    ├── test_cov.py
    ├── test_dft.py
    ├── test_windowed_dft.py
    ├── test_uncertainty.py
    ├── test_validate.py
    ├── test_freq_bt.py
    ├── test_freq_etfe.py
    ├── test_freq_btfdr.py
    ├── test_freq_map.py
    ├── test_spectrogram.py
    ├── test_detrend.py
    ├── test_ltv_disc.py
    ├── test_ltv_disc_tune.py
    ├── test_ltv_disc_var_len.py
    ├── test_ltv_disc_uncertainty.py
    ├── test_ltv_disc_frozen.py
    ├── test_ltv_disc_io.py
    ├── test_ltv_state_est.py
    ├── test_lti_freq_io.py
    ├── test_model_order.py
    ├── test_residual.py
    ├── test_compare.py
    ├── test_plotting.py
    ├── test_multi_trajectory.py
    └── test_cross_validation.py
```

---

## Result Structs

All `sid.freq_*` functions return `FreqResult` (frozen dataclass):

```python
result.frequency          # (nf,) rad/sample
result.frequency_hz       # (nf,) Hz
result.response           # (nf,) or (nf, ny, nu) complex; None for time-series
result.response_std       # same shape, real; NaN for ETFE
result.noise_spectrum     # (nf,) or (nf, ny, ny) real
result.noise_spectrum_std # same shape
result.coherence          # (nf,) squared coherence (SISO only; None for MIMO)
result.sample_time        # scalar (seconds)
result.window_size        # scalar integer (or array for BTFDR)
result.data_length        # N (number of samples used)
result.num_trajectories   # scalar (number of trajectories)
result.method             # 'freq_bt', 'freq_btfdr', 'freq_etfe'
```

---

## Porting Workflow

For each public function, the porting order is:

1. **Port private dependencies** — read MATLAB source, write Python, write tests
2. **Port the public function** — read MATLAB source, write Python with docstring
3. **Port MATLAB tests** — translate test logic to pytest
4. **Write equivalence tests** — load JSON reference data, compare to `rtol=1e-10`
5. **Run full suite** — verify green

This ensures every building block is tested in isolation before composition.

---

## Phased Roadmap

### Phase P1 — Scaffolding ⬜

- `python/pyproject.toml` — packaging (numpy, scipy, optional matplotlib)
- `python/sid/__init__.py` — public API exports
- `python/sid/_results.py` — frozen dataclasses for all result types
- `python/sid/_exceptions.py` — `SidError` exception class
- `python/sid/_internal/__init__.py`
- `python/CONTRIBUTING.md` — Python coding standards and docstring template
- `.github/scripts/check_python_headers.py` — docstring validation
- `.github/workflows/python-lint.yml` — ruff + header check CI
- `python/tests/conftest.py` — shared fixtures, tolerance helpers

### Phase P2 — `freq_bt` SISO Core ⬜

Port the Blackman-Tukey estimator and all its private dependencies.

**Private helpers (in order):**

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 2.1 | `sidValidateData.m` | `_internal/validate_data.py` | `test_validate.py` |
| 2.2 | `sidHannWin.m` | `_internal/hann_win.py` | `test_hann_win.py` |
| 2.3 | `sidCov.m` | `_internal/cov.py` | `test_cov.py` |
| 2.4 | `sidDFT.m` | `_internal/dft.py` | `test_dft.py` |
| 2.5 | `sidIsDefaultFreqs.m` | `_internal/is_default_freqs.py` | (tested via windowed_dft) |
| 2.6 | `sidWindowedDFT.m` | `_internal/windowed_dft.py` | `test_windowed_dft.py` |
| 2.7 | `sidUncertainty.m` | `_internal/uncertainty.py` | `test_uncertainty.py` |

**Public function:**

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 2.8 | `sidFreqBT.m` | `freq_bt.py` | `test_freq_bt.py` |

### Phase P3 — Time Series + MIMO ⬜

Extend `freq_bt` to handle `u=None` (time series mode) and multi-channel
input/output (MIMO). No new files — extends Phase P2 code.

- Time series: `freq_bt(y, None)` returns output spectrum only
- MIMO: matrix-valued covariances, spectral matrix inversion, per-element uncertainty

Tests: additional cases in `test_freq_bt.py`.

### Phase P4 — Multi-Trajectory ⬜

Extend `validate_data`, `cov`, `freq_bt` to accept 3D arrays `(N, nch, L)` and
`list[ndarray]` for variable-length trajectories.

- Ensemble-averaged covariances: `R_ens(tau) = (1/L) sum_l R_l(tau)`
- Variance reduction: `1/(N*L)` in uncertainty formulas
- `num_trajectories` field in output

Tests: `test_multi_trajectory.py`, additional cases in `test_freq_bt.py`.

### Phase P5 — `freq_etfe`, `freq_btfdr`, `detrend` ⬜

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 5.1 | `sidFreqETFE.m` | `freq_etfe.py` | `test_freq_etfe.py` |
| 5.2 | `sidFreqBTFDR.m` | `freq_btfdr.py` | `test_freq_btfdr.py` |
| 5.3 | `sidDetrend.m` | `detrend.py` | `test_detrend.py` |

### Phase P6 — `spectrogram`, `freq_map` ⬜

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 6.1 | `sidSpectrogram.m` | `spectrogram.py` | `test_spectrogram.py` |
| 6.2 | `sidFreqMap.m` | `freq_map.py` | `test_freq_map.py` |

`freq_map` supports two algorithms:
- `algorithm='bt'` (default): calls `freq_bt` per segment
- `algorithm='welch'`: Welch's method per segment

### Phase P7 — `ltv_disc` Core (COSMIC) ⬜

**Private helpers:**

| Step | MATLAB source | Python target |
|------|--------------|---------------|
| 7.1 | `sidTestMSD.m` | `_internal/test_msd.py` |
| 7.2 | `sidLTVbuildDataMatrices.m` | `_internal/ltv_build_data_matrices.py` |
| 7.3 | `sidLTVbuildBlockTerms.m` | `_internal/ltv_build_block_terms.py` |
| 7.4 | `sidLTVcosmicSolve.m` | `_internal/ltv_cosmic_solve.py` |
| 7.5 | `sidLTVevaluateCost.m` | `_internal/ltv_evaluate_cost.py` |

**Public function:**

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 7.6 | `sidLTVdisc.m` | `ltv_disc.py` | `test_ltv_disc.py` |

### Phase P7a — Variable-Length Trajectories ⬜

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 7a.1 | `sidLTVbuildDataMatricesVarLen.m` | extend `ltv_build_data_matrices.py` | `test_ltv_disc_var_len.py` |

### Phase P7b — Bayesian Uncertainty ⬜

**Private helpers:**

| Step | MATLAB source | Python target |
|------|--------------|---------------|
| 7b.1 | `sidLTVuncertaintyBackwardPass.m` | `_internal/ltv_uncertainty_backward_pass.py` |
| 7b.2 | `sidEstimateNoiseCov.m` | `_internal/estimate_noise_cov.py` |
| 7b.3 | `sidExtractStd.m` | `_internal/extract_std.py` |

Tests: `test_ltv_disc_uncertainty.py`

### Phase P7c — `ltv_disc_tune`, `ltv_disc_frozen` ⬜

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 7c.1 | `sidLTVdiscTune.m` | `ltv_disc_tune.py` | `test_ltv_disc_tune.py` |
| 7c.2 | `sidLTVdiscFrozen.m` | `ltv_disc_frozen.py` | `test_ltv_disc_frozen.py` |

### Phase P8 — Output-COSMIC ⬜

Depends on P2 (frequency-domain) and P7 (COSMIC core).

**Private helpers:**

| Step | MATLAB source | Python target |
|------|--------------|---------------|
| 8.1 | `sidLTVblkTriSolve.m` | `_internal/ltv_blk_tri_solve.py` |

**Public functions:**

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 8.2 | `sidLTIfreqIO.m` | `lti_freq_io.py` | `test_lti_freq_io.py` |
| 8.3 | `sidLTVStateEst.m` | `ltv_state_est.py` | `test_ltv_state_est.py` |
| 8.4 | `sidModelOrder.m` | `model_order.py` | `test_model_order.py` |
| 8.5 | `sidLTVdiscIO.m` | `ltv_disc_io.py` | `test_ltv_disc_io.py` |

### Phase P9 — Workflow Utilities ⬜

**Private helpers:**

| Step | MATLAB source | Python target |
|------|--------------|---------------|
| 9.1 | `sidFreqDomainSim.m` | `_internal/freq_domain_sim.py` |

**Public functions:**

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 9.2 | `sidResidual.m` | `residual.py` | `test_residual.py` |
| 9.3 | `sidCompare.m` | `compare.py` | `test_compare.py` |

### Phase P10 — Plotting ⬜

All plotting functions lazy-import matplotlib and accept an optional `ax=`
keyword argument for embedding in user-created figures.

| Step | MATLAB source | Python target | Tests |
|------|--------------|---------------|-------|
| 10.1 | `sidBodePlot.m` | `bode_plot.py` | `test_plotting.py` |
| 10.2 | `sidSpectrumPlot.m` | `spectrum_plot.py` | `test_plotting.py` |
| 10.3 | `sidMapPlot.m` | `map_plot.py` | `test_plotting.py` |
| 10.4 | `sidSpectrogramPlot.m` | `spectrogram_plot.py` | `test_plotting.py` |

### Phase P11 — Cross-Validation + CI ⬜

- `python/tests/test_cross_validation.py` — load JSON from `testdata/`, call
  Python functions, assert `rtol=1e-10`
- `.github/workflows/python-tests.yml` — pytest on Python 3.10–3.13
- Update `.github/workflows/cross-validate.yml` — add `validate-python` job

---

## Dependency Graph

```
P1 (scaffolding)
├── P2 (freq_bt core)
│   ├── P3 (time series + MIMO)
│   │   └── P4 (multi-trajectory)
│   ├── P5 (freq_etfe, freq_btfdr, detrend)
│   │   └── P6 (spectrogram, freq_map)
│   │       └── P10 (plotting)
│   ├── P8 (Output-COSMIC) ←── also needs P7
│   └── P9 (residual, compare) ←── also needs P7
├── P7 (ltv_disc core)
│   ├── P7a (variable-length)
│   ├── P7b (uncertainty)
│   │   └── P7c (tune, frozen)
│   ├── P8 (Output-COSMIC)
│   └── P9 (residual, compare)
└── P11 (cross-validation + CI) ←── after all phases
```

---

## Timeline

| Phase | Description | Dependencies | Status |
|-------|-------------|-------------|--------|
| P1 | Scaffolding | — | ⬜ |
| P2 | `freq_bt` SISO core | P1 | ⬜ |
| P3 | Time series + MIMO | P2 | ⬜ |
| P4 | Multi-trajectory | P3 | ⬜ |
| P5 | `freq_etfe`, `freq_btfdr`, `detrend` | P2 | ⬜ |
| P6 | `spectrogram`, `freq_map` | P2, P5 | ⬜ |
| P7 | `ltv_disc` core (COSMIC) | P1 | ⬜ |
| P7a | Variable-length trajectories | P7 | ⬜ |
| P7b | Bayesian uncertainty | P7 | ⬜ |
| P7c | `ltv_disc_tune`, `ltv_disc_frozen` | P7, P7b | ⬜ |
| P8 | Output-COSMIC | P2, P7 | ⬜ |
| P9 | Workflow utilities | P2, P7 | ⬜ |
| P10 | Plotting | P2, P6 | ⬜ |
| P11 | Cross-validation + CI | all | ⬜ |

---

## Technical Notes

- **FFT convention:** `np.fft.fft` matches MATLAB `fft` (no scaling on forward)
- **1-indexing → 0-indexing:** SPEC formulas are 1-indexed; adjust bin extraction
- **Data layout:** `(N, ny)` same in both; 3D arrays `(N, ny, L)` also match
- **`\` operator:** MATLAB `A\b` → `np.linalg.solve(A, b)`
- **RNG:** MATLAB `rng(42)` differs from `np.random.default_rng(42)` — cross-validation uses stored JSON data, not seeds
- **Plotting:** lazy-import matplotlib; return `dict` of handles; accept `ax=` kwarg
- **Reserved word:** MATLAB `Lambda` → Python `lambda_` (PEP 8 trailing underscore)

---

## Out of Scope (v1.0)

- Online/recursive COSMIC (Phase 8c in MATLAB roadmap — deferred to v2)
- Parametric identification: ARX, ARMAX, N4SID (v2)
- LPV identification (v2)
- Frequency-domain input data
- Continuous-time models
