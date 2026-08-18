# Reproducibility guide

## Environment

Install the Python dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A working Gurobi installation and license are required for optimization. Do not commit `gurobi.lic` or any Web License Service credentials.

## Validation without Gurobi

Run the data-integrity checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

## Suggested run order

```bash
python scripts/solve_case_study.py
python scripts/compare_accident_metrics.py
python scripts/analyze_risk_tradeoffs.py
python scripts/analyze_rerouting.py
python scripts/analyze_cost_components.py
python scripts/run_factorial_analysis.py
python scripts/analyze_new_paths.py
python scripts/run_minmax_regret.py
```

The factorial and regret scripts solve many mixed-integer models and may take substantially longer than the core case-study solve. Generated CSV and PNG files are written to `outputs/`.

## Scope of this release

This is a cleaned research-code archive, not yet a fully refactored Python package. The standalone scripts retain duplicated model-building code to preserve the supplied experiment logic. Refactoring should happen only after the scientific discrepancies in [Implementation audit](implementation_audit.md) are resolved and reference outputs are available for regression testing.

## Verification status

- All included Python files pass syntax compilation.
- Workbook schemas and cross-sheet keys are checked by unit tests.
- Optimization results have not been reproduced in the preparation environment because Gurobi and a solver license were unavailable there.
- Statistical-analysis dependencies are declared but their numerical outputs still require a clean rerun.
