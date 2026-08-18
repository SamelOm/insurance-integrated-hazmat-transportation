# Implementation audit: resolve before public release

This note separates packaging changes from scientific-model changes. No numerical assumption below was silently corrected.

## 1. Active-path allocation constraint

The thesis specifies a minimum active-path share:

$$N_{fp}\geq\theta D_fX_{fp}.$$

The scripts instead enforce:

```python
N[(f, p)] <= PHI * D_trips[f]
```

With `PHI = 0.30`, the implementation imposes a maximum 30% share per path and therefore forces diversification. These are different feasible regions. Confirm which formulation produced the reported results.

## 2. Accident-probability scaling

The workbook stores `P_fp` values between approximately \(1.07\times10^{-7}\) and \(9.93\times10^{-7}\). The thesis uses inconsistent ranges across sections, and the scripts apply experiment-specific multipliers of 1, 5, or 10 to insurance probabilities.

Examples include:

- a factor of 10 in the core premium and uninsured-loss objective;
- a factor of 5 in the rerouting experiment relative to a no-insurance baseline; and
- different factors during threshold, factorial, cost, and regret analyses.

The risk constraints continue to use `R_fp` from the workbook. Confirm whether the multipliers represent a planning horizon, scenario stress, or calibration adjustment, then expose them as named parameters.

## 3. Corridor-cap calibration order

The comments describe a search for the tightest feasible corridor multiplier, but the list is scanned from 1.20 downward and the loop stops at the first feasible value. That typically selects the loosest tested multiplier, not the tightest one.

## 4. Equity multiplier

The code defines `ETA_DELTA = 0.70` and then computes:

```python
HARD_DELTA = max(ETA_DELTA * spread_ab, 0.95 * spread_ab)
```

For any `ETA_DELTA` below 0.95, the result is always 95% of the calibrated spread. The stated 0.70 value has no effect.

## 5. Segment fixed-charge field

The `segments` sheet contains an `F` column, but the scripts use only the lower bound, upper bound, and variable cost. Confirm whether `F` was intentionally excluded.

## 6. Reproduction baseline

The original archive included a serialized `baseline.pkl`. It contained only built-in Python objects and total demand 506,613, but it was removed from this release because serialized pickle files are opaque and the scripts can derive total demand directly from the workbook.

## 7. Required next verification

Before making the repository public:

1. confirm the intended allocation constraint with the advisor;
2. define and document all probability multipliers;
3. correct or relabel the corridor and equity calibration logic;
4. rerun every reported experiment with a clean environment;
5. compare regenerated tables and figures with the thesis; and
6. choose an explicit software license only after ownership is confirmed.
