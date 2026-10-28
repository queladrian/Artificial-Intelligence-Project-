# Estimating Thermal Conductivity from Transient Thermocouple Data Using Physics-Informed Neural Networks

Preliminary study (first version of a research paper in MDPI format). An inverse physics-informed
neural network (PINN) estimates the thermal conductivity `k`, the convective heat transfer
coefficient `h` and the heating power `P` of an aluminum rod from experimental thermocouple data.
The PINN is compared with a published nonlinear least-squares method (baseline).

> **Status: preliminary.** The PINN was evaluated on one test (`50al1`) with one random seed, and its
> parameter estimates were still changing when training stopped. See the paper for the discussion.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/          # CSV files (not redistributed here, see "Data")
├── src/
│   ├── common.py          # parameters of the aluminum rod, data loading
│   ├── baseline_lm.py     # baseline: nonlinear least squares (Levenberg-Marquardt)
│   ├── pinn_inverse.py    # inverse PINN (PyTorch)
│   └── check_physics.py   # finite-difference check of the model formulation
└── results/       # JSON outputs of the experiments
```

## Data

The data are the open dataset by Tomanek and Stutts, released under CC BY 4.0:

- Dataset: L. Tomanek and D. Stutts, *Data on the validation to determine the material thermal
  properties estimation via a one-dimensional transient convection model*, Mendeley Data, V1, 2021,
  DOI: [10.17632/sbcf36976g.1](https://doi.org/10.17632/sbcf36976g.1)
- Data article: L. B. Tomanek and D. S. Stutts, *Data in Brief* 40 (2022) 107632,
  DOI: [10.1016/j.dib.2021.107632](https://doi.org/10.1016/j.dib.2021.107632)

This study uses three tests of the Al6061-T6 rod at a wind speed of 5 m/s: `50al1`, `75al1`
and `100al1` (boundary temperatures of 50, 75 and 100 °C). The raw files are `.xls`; the only
modification made is the conversion to `.csv`.

1. Download the dataset from the DOI above and unzip it (the raw files are in the `Raw data` folder).
2. Convert the three files to CSV (run from the repository root, with the unzipped folder next to it):

```python
import pandas as pd
for name in ["50al1", "75al1", "100al1"]:
    pd.read_excel(f"Raw data/{name}.xls").to_csv(f"data/{name}.csv", index=False)
```

Each CSV has the columns `Time`, `TC1`–`TC6` (excess temperature in °C, ordered from the heater
outward) and `Power`.

## Installation

Python 3.11 or 3.12 is recommended (PyTorch may not support newer versions).

```
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the experiments

Run all commands from the repository root.

```
# 1. Check that the model formulation matches the analytical solution (finite differences)
python src/check_physics.py

# 2. Baseline (one command per test)
python src/baseline_lm.py data/50al1.csv  > results/baseline_50al1.json
python src/baseline_lm.py data/75al1.csv  > results/baseline_75al1.json
python src/baseline_lm.py data/100al1.csv > results/baseline_100al1.json

# 3. Inverse PINN
python src/pinn_inverse.py data/50al1.csv --iters 40000 --seed 0
```

The PINN writes `results/pinn_50al1_seed0.json`. Training progress is printed every 1000 iterations.
The baseline prints a small `RuntimeWarning` (invalid value in sqrt) while the optimizer explores
parameter values; it does not affect the final result.

## Experimental configuration

| Setting | Value |
|---|---|
| Rod | Al6061-T6, D = 3.175 mm, L = 150 mm, rho = 2767.99 kg/m³, c = 896 J/(kg·K) |
| Test conditions | T∞ = 20.5 °C, wind speed 5 m/s |
| Thermocouple positions | 0, 10, 20, 30, 40, 50 mm |
| PINN architecture | 3 hidden layers, 64 neurons, tanh |
| Optimizer | Adam; learning rate 1e-3 (network), 1e-2 (k, h, P) |
| Iterations | 40 000 |
| Loss weights (data, PDE, x=0, x=L, IC) | 10, 1, 1, 1, 10 |
| Initial values of k, h, P | 100, 130.8, 1.0 |
| Random seed | 0 |

The learning-rate schedule (`StepLR`) is defined in `src/pinn_inverse.py`; check the `gamma` value
used in each run.

## Preliminary results

Baseline (nonlinear least squares):

| Test | k (W/m·K) | h (W/m²·K) | RMSE (°C) |
|---|---|---|---|
| 50al1 | 146.5 | 141.5 | 0.38 |
| 75al1 | 153.7 | 139.1 | 0.67 |
| 100al1 | 148.7 | 139.0 | 1.07 |

Baseline vs. inverse PINN on `50al1`:

| Method | k | h | P (W) | RMSE (°C) |
|---|---|---|---|---|
| Baseline | 146.5 | 141.5 | 1.42 | 0.377 |
| PINN (40 000 iterations) | 126.8 | 119.1 | 1.20 | 0.381 |

The PINN reproduces the measured temperatures with an error comparable to the baseline, but its
estimates of k, h and P are about 13–16 % lower. The baseline is itself an estimate, not a ground
truth.

## Limitations

- One test and one random seed for the PINN; parameters had not converged at the end of training.
- The baseline is a reimplementation of the published procedure and does not compute parameter
  uncertainty.
- Not yet compared with the reference values reported by the dataset authors or with nominal
  material properties.

## Acknowledgments and licenses

- Data: Tomanek and Stutts (CC BY 4.0). Files were converted from `.xls` to `.csv`.
- The baseline reimplements the estimation procedure of the `oneDkhEstimator` code
  (L. B. Tomanek and D. S. Stutts, Zenodo, DOI: 10.5281/zenodo.5683312, MIT license) with the
  parameters of the aluminum rod.
- Method: L. B. Tomanek and D. S. Stutts, *Applied Thermal Engineering* 184 (2021),
  DOI: 10.1016/j.applthermaleng.2020.116362.
