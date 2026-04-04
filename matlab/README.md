# sid — MATLAB/Octave Implementation

MATLAB/Octave implementation of the sid system identification toolbox.

## Requirements

- MATLAB R2016b+ or GNU Octave 8.0+
- No toolboxes required (pure MATLAB/Octave code)

## Installation

```matlab
% Add sid to the MATLAB/Octave path (run once per session)
run('/path/to/sid/matlab/sid/sidInstall.m')

% Or add to startup.m for permanent installation
```

## Quick Start

```matlab
% SISO frequency response estimation
N = 1000; u = randn(N, 1);
y = filter([1], [1 -0.9], u) + 0.1*randn(N, 1);
result = sidFreqBT(y, u);
sidBodePlot(result);

% Time series spectrum
result = sidFreqBT(y, []);
sidSpectrumPlot(result);

% LTV state-space identification
result = sidLTVdisc(X, U, 'Lambda', 1e5);
```

## Testing

```matlab
run('matlab/tests/runAllTests.m')
run('matlab/examples/runAllExamples.m')
```

## Documentation

- [Algorithm specification](../spec/SPEC.md)
- [Examples guide](examples/README.md)
- [Development roadmap](../docs/roadmap.md)
