# Cross-Language Reference Data

This directory contains canonical test vectors used to verify numerical
equivalence across the MATLAB, Python, and Julia implementations of sid.

## Generating reference data

Run from the repository root in MATLAB:

```matlab
run('testdata/generate_reference.m')
```

This produces JSON files with full double-precision outputs for a curated
set of test cases. Each JSON file contains:

- `function` — the sid function name
- `params` — algorithm parameters used
- `input` — input data arrays (not seeds, to avoid RNG differences)
- `output` — expected outputs at 16 significant digits
- `tolerance` — per-field relative tolerances for cross-language comparison

## Format

All numeric arrays are stored as flat JSON arrays of numbers formatted with
`sprintf('%.16e', ...)` to preserve full double precision. Complex arrays
are split into `_real` and `_imag` components.
