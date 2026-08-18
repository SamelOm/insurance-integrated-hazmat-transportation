# Model summary

## Decision setting

Each facility \(f\in F\) has demand \(D_f\) and a set of precomputed paths \(P_f\). The implementation chooses active paths, assigns integer trips, and selects one cost segment for every used facility-path pair.

## Main variables

| Variable | Meaning |
| --- | --- |
| \(X_{fp}\) | 1 when path \(p\) is activated for facility \(f\) |
| \(N_{fp}\) | Integer trips allocated to facility-path pair \((f,p)\) |
| \(y_{fps}\) | 1 when cost segment \(s\) is selected |
| \(N_{fps}\) | Trips assigned to segment \(s\) |
| \(z_{\max},z_{\min}\) | Maximum and minimum regional risk |

## Objective

The thesis formulation minimizes

$$
\sum_{f\in F}\sum_{p\in P_f}
\left[
U_{fp}X_{fp}
+\sum_{s\in T}c_sN_{fps}
+P_{fp}A(0)N_{fp}
+\lambda P_{fp}C^{\mathrm{firm}}_{fp}N_{fp}
\right],
$$

where retained loss is

$$
C^{\mathrm{firm}}_{fp}=\min(S_{fp},D)+\max(0,S_{fp}-L).
$$

The supplied implementation applies additional probability multipliers in several experiments. These are documented in [Implementation audit](implementation_audit.md) and should be confirmed before public release.

## Core constraints

The model enforces:

1. demand satisfaction for every facility;
2. linking between path activation and flow;
3. one bulk-pricing segment per active path;
4. a system-wide risk budget \(\alpha\);
5. corridor-specific risk caps \(\beta_k\); and
6. regional risk disparity no greater than \(\delta\).

The thesis includes a minimum active-path share \(N_{fp}\geq\theta D_fX_{fp}\). The code instead limits each path to at most `PHI * D_f`. This is a substantive modeling difference, not a naming issue.

## Risk construction

- Path risk: \(R_{fp}=P_{fp}E_{fp}\).
- Corridor contributions are normalized in code so that \(\sum_kR_{fpk}=R_{fp}\).
- Regional contributions in the supplied workbook already satisfy \(\sum_mR_{fpm}=R_{fp}\) to numerical tolerance.
- Severity ranks paths by risk and assigns a lognormal body plus a 5% Pareto tail.
