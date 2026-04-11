# sid Examples

Runnable scripts demonstrating the full functionality of the sid toolbox.
Each example is self-contained and can be run directly in MATLAB or Octave.

## Why spring-mass-damper plants?

Every example in this directory is built around a **spring-mass-damper (SMD)
chain** — one, two, or three lumped masses connected by linear springs and
viscous dampers to a wall. The discrete-time state-space model is built with
`util_msd` (exact zero-order-hold via matrix exponential), its time-varying
variant `util_msd_ltv`, or the nonlinear Duffing simulator `util_msd_nl`, all
defined as sibling files in this directory.

Using a single physical plant family across the tutorials keeps the narrative
concrete and cumulative. Instead of arbitrary `filter` coefficients or
hand-crafted state-space matrices, the reader works through a progression of
plants they can picture: a 1-DoF oscillator, a 2-mass chain with cross-coupling
through a shared spring, a 3-mass chain with three closely-spaced modes, a
structure whose stiffness drifts thermally, a structure that snaps, and
finally a Duffing-style nonlinear oscillator whose apparent resonance depends
on amplitude.

The binding specification for every example is [`spec/EXAMPLES.md`](../../spec/EXAMPLES.md).
These MATLAB scripts are the MATLAB port of that specification; the Python
Jupyter notebooks in `python/examples/` are the Python port. Both ports share
the same plant parameters, section order, and `sid*` function call graph.

## Example Index

| Example | Plant | Functions demonstrated |
|---|---|---|
| [`exampleSISO`](exampleSISO.m) | 1-DoF SDOF (`m=1, k=100, c=2`, `ω_n=10 rad/s`, `ζ=0.1`) | `sidFreqBT`, `sidBodePlot`, `sidSpectrumPlot`, `sidDetrend`, `sidResidual`, `sidCompare` |
| [`exampleETFE`](exampleETFE.m) | Same 1-DoF SDOF as `exampleSISO` | `sidFreqETFE`, `sidBodePlot`, `sidSpectrumPlot` |
| [`exampleFreqDepRes`](exampleFreqDepRes.m) | 3-mass chain (`k=[300,200,100]`, modes at 6.4 / 15.1 / 25.1 rad/s) | `sidFreqBT`, `sidFreqBTFDR`, `sidBodePlot` |
| [`exampleCoherence`](exampleCoherence.m) | 2-mass chain with colored force disturbance at mass 2 | `sidFreqBT`, `sidBodePlot` (with coherence) |
| [`exampleMethodComparison`](exampleMethodComparison.m) | Same 1-DoF SDOF as `exampleSISO` | `sidFreqBT`, `sidFreqBTFDR`, `sidFreqETFE`, `sidCompare` |
| [`exampleMIMO`](exampleMIMO.m) | 2-mass chain, 2 force inputs and 2 position outputs | `sidFreqBT` (MIMO mode) |
| [`exampleFreqMap`](exampleFreqMap.m) | 2-mass LTI + continuous LTV (`k₁` ramp) + discrete LTV (step change) + Duffing hardening SDOF | `sidFreqMap`, `sidMapPlot` |
| [`exampleSpectrogram`](exampleSpectrogram.m) | SDOF (`ω_n ≈ 32 Hz`, `Fs=1000 Hz`) driven by a 20→60 Hz chirp force | `sidSpectrogram`, `sidSpectrogramPlot` |
| [`exampleLTVdisc`](exampleLTVdisc.m) | 1-DoF LTV (ramping stiffness) **plus** Duffing linearization | `sidLTVdisc`, `sidLTVdiscTune`, `sidLTVdiscFrozen`, `sidCompare`, `sidResidual` |
| [`exampleMultiTrajectory`](exampleMultiTrajectory.m) | 2-mass ensemble + step-change LTV + chirp-response spectrogram + LTV / `sidFreqMap` consistency | `sidFreqBT`, `sidFreqMap`, `sidSpectrogram`, `sidLTVdisc` (all ensemble mode) |
| [`exampleOutputCOSMIC`](exampleOutputCOSMIC.m) | 2-mass chain, position-only measurements (velocities hidden) | `sidFreqBT`, `sidModelOrder`, `sidLTVdiscIO` |

## Helper API

The three `util_msd*` helpers are specified in
[`spec/EXAMPLES.md` §2](../../spec/EXAMPLES.md) and implemented in:

- [`util_msd.m`](util_msd.m) — exact ZOH discretization of an n-mass chain.
- [`util_msd_ltv.m`](util_msd_ltv.m) — per-step `(Ad(k), Bd(k))` stack for an
  n-mass chain with parameters that vary over the record.
- [`util_msd_nl.m`](util_msd_nl.m) — RK4 simulator for an n-mass chain with
  Duffing-style cubic stiffness.

## Running All Examples

Run the full suite from MATLAB or Octave:

```matlab
run('matlab/examples/runAllExamples.m')
```

The runner discovers every `example*.m` file by glob (no hardcoded manifest),
adds the `sid/` functions to the path, runs each example, and reports the
number of completed sections per file. This runner is also used in CI to
validate examples on both MATLAB and GNU Octave.

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for MATLAB function header
conventions, example-script templates, and auto-discovery rules. New examples
must conform to the binding specification in
[`../../spec/EXAMPLES.md`](../../spec/EXAMPLES.md) — add a new entry there
first (minor version bump), then implement both Python and MATLAB versions.
