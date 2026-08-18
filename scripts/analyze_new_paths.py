"""Analyze newly activated paths across the factorial experiment."""
import itertools
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# -------------------------
# USER SETTINGS
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "master_data.xlsx"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Diversification / anti-dumping (max share on any one path)
PHI = 0.3

# Insurance parameters
A0 = 100_000_000
LAMBDA = 200
DEDUCTIBLE = 10_000.0
LIMIT_L = 1_000_000.0

# Calibration multipliers (tightening from baseline)
ETA_ALPHA = 1.5
ETA_DELTA = 0.70

SEED = 42

# Corridor handling:
OTHER_ID = "9"            # corridor_id used for the uncategorized remainder
CAP_OTHER = False         # True => enforce a beta cap for OTHER too
BETA_OTHER_VALUE = 1e9    # only used if CAP_OTHER=True

# -------------------------
# HELPERS: C_firm from severity
# -------------------------
def build_S_fp_continuous_from_paths_df(
    paths_df: pd.DataFrame,
    risk_col: str = "R_fp",
    facility_col: str = "facility_id",
    path_col: str = "path_id",
    tie_breaker_col: str = "U_fp",
    S_min: float = 1e5,
    S_med: float = 3e5,
    top_tail_share: float = 0.05,
    tail_min: float = 1e6,
    tail_max: float = 5e6,
    pareto_alpha: float = 2.5,
    seed: int = 42,
):
    df = paths_df.copy()
    df[risk_col] = pd.to_numeric(df[risk_col], errors="coerce")
    if df[risk_col].isna().any():
        bad = df[df[risk_col].isna()][[facility_col, path_col]].head(10)
        raise ValueError(f"Blank/non-numeric {risk_col}. Examples:\n{bad.to_string(index=False)}")

    n = len(df)
    rng = np.random.default_rng(seed)

    # deterministic jitter for stable ranking
    if tie_breaker_col in df.columns:
        tb = pd.to_numeric(df[tie_breaker_col], errors="coerce").fillna(0.0).to_numpy()
        jitter = 1e-12 * (tb - tb.mean())
    else:
        jitter = 1e-12 * rng.standard_normal(n)

    score = df[risk_col].to_numpy() + jitter
    asc_idx = np.argsort(score)  # low risk -> high risk

    n_tail = int(np.ceil(top_tail_share * n))
    n_body = n - n_tail
    assigned = np.empty(n, dtype=float)

    # BODY
    if n_body > 0:
        sigma = 0.65
        mu = np.log(S_med)
        normal_samples = rng.standard_normal(200000)
        normal_samples.sort()

        body_idx = asc_idx[:n_body]
        q = np.linspace(0.02, 0.98, n_body)
        z_idx = (q * (len(normal_samples) - 1)).astype(int)
        z = normal_samples[z_idx]
        s_body = np.exp(mu + sigma * z)
        s_body = np.clip(s_body, S_min, max(S_min, tail_min * 0.999))
        assigned[body_idx] = s_body

    # TAIL
    if n_tail > 0:
        tail_idx = asc_idx[n_body:]
        y = rng.pareto(pareto_alpha, size=n_tail)
        x = 1.0 + y
        s_tail = tail_min * x
        s_tail = np.clip(s_tail, tail_min, tail_max)
        s_tail.sort()
        assigned[tail_idx] = s_tail

    df["S_fp"] = assigned
    return {(row[facility_col], row[path_col]): float(row["S_fp"]) for _, row in df.iterrows()}


def derive_Cfirm(paths_df: pd.DataFrame, S_fp: dict):
    C_firm = {}
    for _, row in paths_df.iterrows():
        f = row["facility_id"]
        p = row["path_id"]
        s_val = float(S_fp[(f, p)])
        c_firm = min(s_val, DEDUCTIBLE) + max(0.0, s_val - LIMIT_L)
        C_firm[(f, p)] = float(c_firm)
    return C_firm


# -------------------------
# LOAD DATA
# -------------------------
print("Loading Excel:", DATA_FILE)

fac  = pd.read_excel(DATA_FILE, sheet_name="facilities")
pth  = pd.read_excel(DATA_FILE, sheet_name="paths")
seg  = pd.read_excel(DATA_FILE, sheet_name="segments")
corr = pd.read_excel(DATA_FILE, sheet_name="corridors")
r_fpk_df = pd.read_excel(DATA_FILE, sheet_name="corridor_path_risk")
r_fpm_df = pd.read_excel(DATA_FILE, sheet_name="region_path_risk")

# normalize ids as strings (critical: consistent keys everywhere)
fac["facility_id"] = fac["facility_id"].astype(str)
pth["facility_id"] = pth["facility_id"].astype(str)
pth["path_id"]     = pth["path_id"].astype(str)

corr["corr_id"] = corr["corr_id"].astype(str)

r_fpk_df["facility_id"] = r_fpk_df["facility_id"].astype(str)
r_fpk_df["path_id"]     = r_fpk_df["path_id"].astype(str)
r_fpk_df["corridor_id"] = r_fpk_df["corridor_id"].astype(str)

r_fpm_df["facility_id"] = r_fpm_df["facility_id"].astype(str)
r_fpm_df["path_id"]     = r_fpm_df["path_id"].astype(str)
r_fpm_df["region_id"]   = r_fpm_df["region_id"].astype(str)

# Sets
F = fac["facility_id"].unique().tolist()
M = r_fpm_df["region_id"].unique().tolist()

# Paths per facility
P = {f: pth.loc[pth["facility_id"] == f, "path_id"].unique().tolist() for f in F}

# Demand
D_trips = {row["facility_id"]: int(row["D_f"]) for _, row in fac.iterrows()}

# Path params
U_fp = {(row["facility_id"], row["path_id"]): float(row["U_fp"]) for _, row in pth.iterrows()}
R_fp = {(row["facility_id"], row["path_id"]): float(row["R_fp"]) for _, row in pth.iterrows()}

# Use the accident probabilities supplied in the workbook
if "P_fp" not in pth.columns:
    raise ValueError("paths sheet is missing the required P_fp column.")
P_fp = {(row["facility_id"], row["path_id"]): float(row["P_fp"]) for _, row in pth.iterrows()}

# Corridor risk attribution (keys use corridor_id from table)
R_fpk = {(row["facility_id"], row["path_id"], row["corridor_id"]): float(row["R_fpk"])
         for _, row in r_fpk_df.iterrows()}

# Region risk attribution
R_fpm = {(row["facility_id"], row["path_id"], row["region_id"]): float(row["R_fpm"])
         for _, row in r_fpm_df.iterrows()}
# ---- Corridor sets
K_all = sorted(r_fpk_df["corridor_id"].unique().tolist())          # includes "9" if present
K_tracked = sorted(corr["corr_id"].unique().tolist())             # should be "1".."8"

missing_in_corr = set(K_all) - set(K_tracked)
if missing_in_corr:
    print("Note: corridor ids in corridor_path_risk not in corridors sheet:", missing_in_corr)

# We'll cap these corridors:
K_cap = K_tracked + ([OTHER_ID] if CAP_OTHER and (OTHER_ID in K_all) else [])

# ---- Bulk pricing tiers
seg["s"] = seg["s"].astype(int)
T = sorted(seg["s"].unique().tolist())

L_seg = {int(r.s): float(r.L) for _, r in seg.iterrows()}
U_seg = {int(r.s): float(r.U) for _, r in seg.iterrows()}
c_ops = {int(r.s): float(r.c) for _, r in seg.iterrows()}

# Open-ended top tier: cap by max facility demand
U_MAX = max(D_trips.values())
TOP_S = max(U_seg.keys())
U_seg[TOP_S] = U_MAX

# ---- Build severity + C_firm (P_fp stays from file)
S_fp = build_S_fp_continuous_from_paths_df(pth, seed=SEED)
C_firm = derive_Cfirm(pth, S_fp)
ins_coeffs = []
ops_coeffs = []

for f in F:
    for p in P[f]:
        ins =  P_fp[(f,p)]*A0 + LAMBDA*P_fp[(f,p)]*C_firm[(f,p)]
        ins_coeffs.append(ins)

        for s in T:
            ops_coeffs.append(c_ops[s])

print("Insurance coefficient range:", min(ins_coeffs), max(ins_coeffs))
print("Operational variable cost range:", min(ops_coeffs), max(ops_coeffs))
# ===============================
# NORMALIZE corridor risk shares
# Option B:
# Keep overlap-based corridor contributions, but force
# sum over corridor_id of R_fpk = R_fp for each (facility_id, path_id)
# ===============================
eps = 1e-12
normalized_count = 0
zero_raw_count = 0

R_fpk_raw = R_fpk.copy()
R_fpk = {}

for facility_id in F:
    for path_id in P[facility_id]:
        total_path_risk = R_fp[(facility_id, path_id)]

        # collect all raw corridor contributions for this path
        raw_vals = {
            corridor_id: R_fpk_raw.get((facility_id, path_id, corridor_id), 0.0)
            for corridor_id in K_all
        }

        raw_sum = sum(raw_vals.values())

        if raw_sum <= eps:
            # no corridor attribution found for this path
            for corridor_id in K_all:
                R_fpk[(facility_id, path_id, corridor_id)] = 0.0
            zero_raw_count += 1
        else:
            for corridor_id in K_all:
                R_fpk[(facility_id, path_id, corridor_id)] = (
                    total_path_risk * raw_vals[corridor_id] / raw_sum
                )
            normalized_count += 1

print(f"Normalized corridor risk for {normalized_count} paths")
print(f"Paths with zero raw corridor attribution: {zero_raw_count}")
# ===============================
# CHECK: Corridor risk consistency
# ===============================
# ===============================
# CHECK: normalized corridor risk consistency
# After normalization, for each (facility_id, path_id),
# sum over corridor_id of R_fpk should equal R_fp
# ===============================
bad = 0
max_ratio = 0.0

for facility_id in F:
    for path_id in P[facility_id]:
        corridor_sum = sum(
            R_fpk.get((facility_id, path_id, corridor_id), 0.0)
            for corridor_id in K_all
        )
        total_path_risk = R_fp[(facility_id, path_id)]

        if abs(corridor_sum - total_path_risk) > 1e-8 * max(1.0, total_path_risk):
            bad += 1
            if total_path_risk > 0:
                max_ratio = max(max_ratio, corridor_sum / total_path_risk)

print("Paths where sum over corridor_id of R_fpk != R_fp:", bad)
print("Max ratio (sum R_fpk / R_fp):", max_ratio)

# -------------------------
# MODEL BUILDER
# -------------------------
def build_model(
    HARD_ALPHA,
    beta_k,
    HARD_DELTA,
    enforce_equity: bool,
    P_fp_use=None,
    C_firm_use=None,
    use_insurance=True,
    A0_val=None,
    LAMBDA_val=None
):
    m = Model("hazmat_thesis")
    m.Params.OutputFlag = 1
    m.Params.Seed = SEED
    m.Params.NumericFocus = 2

    m.Params.MIPGap = 0.005
    if P_fp_use is None:
        P_fp_use = P_fp

    # Variables
    X = {(f, p): m.addVar(vtype=GRB.BINARY,  name=f"X[{f},{p}]") for f in F for p in P[f]}
    N = {(f, p): m.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"N[{f},{p}]") for f in F for p in P[f]}

    y = {(f, p, s): m.addVar(vtype=GRB.BINARY,  name=f"y[{f},{p},{s}]")    for f in F for p in P[f] for s in T}
    N_seg = {(f, p, s): m.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"Nseg[{f},{p},{s}]") for f in F for p in P[f] for s in T}

    # Objective
    # Objective
    obj_terms = []
    for f in F:
        for p in P[f]:
            ops = U_fp[(f, p)] * X[(f, p)] + quicksum(c_ops[s] * N_seg[(f, p, s)] for s in T)

            if use_insurance:
                A0_use = A0_val if A0_val is not None else A0
                LAMBDA_use = LAMBDA_val if LAMBDA_val is not None else LAMBDA
                Cfirm_use = C_firm_use if C_firm_use is not None else C_firm

                prem = P_fp_use[(f, p)] * A0_use * N[(f, p)]
                unins = LAMBDA_use * P_fp_use[(f, p)] * Cfirm_use[(f, p)] * N[(f, p)]
            else:
                prem = 0.0
                unins = 0.0

            obj_terms.append(ops + prem + unins)

    m.setObjective(quicksum(obj_terms), GRB.MINIMIZE)

    # Constraints
    for f in F:
        m.addConstr(quicksum(N[(f, p)] for p in P[f]) == D_trips[f], name=f"Demand[{f}]")

        for p in P[f]:
            m.addConstr(N[(f, p)] <= D_trips[f] * X[(f, p)], name=f"LinkUB[{f},{p}]")
            m.addConstr(N[(f, p)] <= PHI * D_trips[f],       name=f"MaxShare[{f},{p}]")

            # Bulk pricing: exactly one tier if path used
            m.addConstr(quicksum(y[(f, p, s)] for s in T) == X[(f, p)], name=f"OneSeg[{f},{p}]")
            m.addConstr(quicksum(N_seg[(f, p, s)] for s in T) == N[(f, p)], name=f"SegFlow[{f},{p}]")

            for s in T:
                m.addConstr(N_seg[(f, p, s)] >= L_seg[s] * y[(f, p, s)], name=f"SegLB[{f},{p},{s}]")
                m.addConstr(N_seg[(f, p, s)] <= U_seg[s] * y[(f, p, s)], name=f"SegUB[{f},{p},{s}]")

    # Risk constraints
    sys_risk = quicksum(R_fp[(f, p)] * N[(f, p)] for f in F for p in P[f])
    m.addConstr(sys_risk <= HARD_ALPHA, name="SystemRisk")

    for k in K_cap:
        k_risk = quicksum(R_fpk.get((f, p, k), 0.0) * N[(f, p)] for f in F for p in P[f])
        m.addConstr(k_risk <= beta_k[k], name=f"Cap[{k}]")

    # Equity
    z_max = m.addVar(vtype=GRB.CONTINUOUS, name="z_max")
    z_min = m.addVar(vtype=GRB.CONTINUOUS, name="z_min")
    if enforce_equity:
        for reg in M:
            r_expr = quicksum(R_fpm.get((f, p, reg), 0.0) * N[(f, p)] for f in F for p in P[f])
            m.addConstr(r_expr <= z_max, name=f"RegUB[{reg}]")
            m.addConstr(r_expr >= z_min, name=f"RegLB[{reg}]")
        m.addConstr(z_max - z_min <= HARD_DELTA, name="Equity")

    m._X = X
    m._N = N
    m._zmax = z_max
    m._zmin = z_min
    m._Nseg = N_seg
    return m


def compute_risk_metrics(m):
    N = m._N

    R_base = sum(R_fp[(f, p)] * N[(f, p)].X for f in F for p in P[f])

    Rk_base = {}
    for k in K_all:
        Rk_base[k] = sum(R_fpk.get((f, p, k), 0.0) * N[(f, p)].X for f in F for p in P[f])

    Rm = {}
    for reg in M:
        Rm[reg] = sum(R_fpm.get((f, p, reg), 0.0) * N[(f, p)].X for f in F for p in P[f])

    spread = (max(Rm.values()) - min(Rm.values())) if len(Rm) else 0.0
    return R_base, Rk_base, spread


def objective_breakdown(m, P_fp_use=None):
    if P_fp_use is None:
        P_fp_use = P_fp

    X, N, Nseg = m._X, m._N, m._Nseg

    ops_fixed = sum(U_fp[(f, p)] * X[(f, p)].X for f in F for p in P[f])
    ops_var   = sum(c_ops[s] * Nseg[(f, p, s)].X for f in F for p in P[f] for s in T)
    ops_total = ops_fixed + ops_var

    prem_total = sum(P_fp_use[(f, p)] * A0 * N[(f, p)].X for f in F for p in P[f])
    ship_total = sum(LAMBDA * P_fp_use[(f, p)] * C_firm[(f, p)] * N[(f, p)].X for f in F for p in P[f])

    total = ops_total + prem_total + ship_total
    return ops_total, prem_total, ship_total, total
results = []

def run_and_store(exp_name, param_name, param_value, alpha_val, beta_val, delta_val):
    m = build_model(
        HARD_ALPHA=alpha_val,
        beta_k=beta_val,
        HARD_DELTA=delta_val,
        enforce_equity=True
    )
    m.optimize()

    if m.status != GRB.OPTIMAL:
        results.append({
            "experiment": exp_name,
            "param": param_name,
            "value": param_value,
            "status": m.status
        })
        return

    ops, prem, ship, total = objective_breakdown(m)

    sys_used = sum(R_fp[(f, p)] * m._N[(f, p)].X for f in F for p in P[f])
    sys_util = sys_used / alpha_val if alpha_val > 0 else np.nan

    corr_utils = []
    for k in K_cap:
        used = sum(R_fpk.get((f, p, k), 0.0) * m._N[(f, p)].X for f in F for p in P[f])
        cap = beta_val[k]
        corr_utils.append(used / cap if cap > 0 else np.nan)

    max_corr_util = max(corr_utils) if corr_utils else np.nan

    active_paths = sum(1 for f in F for p in P[f] if m._X[(f, p)].X > 0.5)

    results.append({
        "experiment": exp_name,
        "param": param_name,
        "value": param_value,
        "status": m.status,
        "objective": m.objVal,
        "ops_cost": ops,
        "premium_cost": prem,
        "uninsured_cost": ship,
        "sys_risk": sys_used,
        "sys_risk_util": sys_util,
        "max_corr_util": max_corr_util,
        "active_paths": active_paths
    })
def generate_scenario(seed):
    rng = np.random.default_rng(seed)

    # scenario-specific accident probabilities: +/- 20%
    P_fp_scn = {
        (f, p): P_fp[(f, p)] * rng.uniform(0.8, 1.2)
        for f in F for p in P[f]
    }

    # scenario-specific severity and resulting firm loss
    S_fp_scn = build_S_fp_continuous_from_paths_df(pth, seed=seed)
    C_firm_scn = derive_Cfirm(pth, S_fp_scn)

    return P_fp_scn, C_firm_scn
# -------------------------
# CALIBRATION LADDER
# -------------------------
print("\n=== CALIBRATION LADDER ===")
HUGE = 1e9

# ---- (A) BASELINE: loose alpha, loose beta, no equity
beta_loose = {k: HUGE for k in K_cap}
m0 = build_model(HARD_ALPHA=HUGE, beta_k=beta_loose, HARD_DELTA=HUGE, enforce_equity=False)
m0.optimize()
if m0.status != GRB.OPTIMAL:
    raise RuntimeError("Baseline infeasible. Check the data, segment bounds, path availability, and PHI.")

R_base, Rk_base, spread_base = compute_risk_metrics(m0)
print(f"Baseline system risk: {R_base:.6f}")

# ---- (B) SET ALPHA from baseline, keep beta loose
HARD_ALPHA = ETA_ALPHA * R_base
mA = build_model(HARD_ALPHA=HARD_ALPHA, beta_k=beta_loose, HARD_DELTA=HUGE, enforce_equity=False)
mA.optimize()
if mA.status != GRB.OPTIMAL:
    raise RuntimeError("Alpha-only infeasible. Loosen ETA_ALPHA or PHI.")

_, Rk_alpha, _ = compute_risk_metrics(mA)

# ---- (C) Find tightest feasible ETA_BETA (monotone scan)
etas = [1.20, 1.10, 1.05, 1.02, 1.00, 0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
best_beta = None
best_eta = None

for eta in etas:
    beta_try = {k: max(1e-6, eta * Rk_alpha.get(k, 0.0)) for k in K_tracked}
    if CAP_OTHER and (OTHER_ID in K_all):
        beta_try[OTHER_ID] = BETA_OTHER_VALUE

    m1 = build_model(HARD_ALPHA=HARD_ALPHA, beta_k=beta_try, HARD_DELTA=HUGE, enforce_equity=False)
    m1.optimize()
    print(f"ETA_BETA={eta:.2f} status={m1.status}")
    if m1.status == GRB.OPTIMAL:
        best_beta = beta_try
        best_eta = eta
        break

if best_beta is None:
    raise RuntimeError("No feasible beta found. Try loosening PHI or using higher ETA_BETA values.")

beta_k = best_beta
print("Chosen ETA_BETA:", best_eta)

# ---- (D) Alpha+beta solve (no equity) to get spread for delta
m1 = build_model(HARD_ALPHA=HARD_ALPHA, beta_k=beta_k, HARD_DELTA=HUGE, enforce_equity=False)
m1.optimize()
if m1.status != GRB.OPTIMAL:
    raise RuntimeError("Alpha+Beta stage infeasible. Loosen ETA_BETA or PHI.")

_, _, spread_ab = compute_risk_metrics(m1)

# ---- (E) Set delta from alpha+beta solution, then final solve with equity
HARD_DELTA = max(ETA_DELTA * spread_ab, 0.95 * spread_ab)  # floor for feasibility

print("\n=== CALIBRATED VALUES ===")
print(f"HARD_ALPHA = {HARD_ALPHA:.6f}")
print(f"HARD_DELTA = {HARD_DELTA:.6f}")
print("beta_k (capped corridors):")
for k in K_cap:
    print(f"  {k}: {beta_k[k]:.6f}")

# ===============================
# THRESHOLD DETECTION (1 instance)
# ===============================

print("\n=== THRESHOLD SCAN ===")

test_s = 0
P_fp_scn, C_firm_scn = generate_scenario(seed=42 + test_s)



A0_grid = np.linspace(5e6, 40e6, 25)

switch_points = []

prev_pattern = None

for A0_val in A0_grid:

    m = build_model(
        HARD_ALPHA=HARD_ALPHA,
        beta_k=beta_k,
        HARD_DELTA=HARD_DELTA,
        enforce_equity=True,
        use_insurance=True,
        A0_val=A0_val,
        LAMBDA_val=200,
        P_fp_use={k: 10 * v for k, v in P_fp_scn.items()},
        C_firm_use=C_firm_scn
    )

    m.Params.OutputFlag = 0
    m.optimize()

    pattern = tuple((f,p) for f in F for p in P[f] if m._X[(f,p)].X > 0.5)

    if prev_pattern is not None and pattern != prev_pattern:
        print("Switch at A0 ≈", A0_val)
        switch_points.append(A0_val)

    prev_pattern = pattern
# ===============================
# EXPERIMENT DESIGN
# ===============================

A0_levels = [1_000_000,7_000_000, 10_000_000]
LAMBDA_levels = [15, 30, 50]

NUM_INSTANCES = 12

exp_results = []

total_shipments = sum(D_trips.values())


# ===============================
# MAIN LOOP: instances × configs
# ===============================

for s in range(NUM_INSTANCES):

    print(f"\n===== INSTANCE {s} =====")

    P_fp_scn, C_firm_scn = generate_scenario(seed=42 + s)

    flows = {}

    # ---- RUN ALL CONFIGS
    for A0_val, L_val in itertools.product(A0_levels, LAMBDA_levels):

        m = build_model(
            HARD_ALPHA=HARD_ALPHA,
            beta_k=beta_k,
            HARD_DELTA=HARD_DELTA,
            enforce_equity=True,
            use_insurance=True,
            A0_val=A0_val,
            LAMBDA_val=L_val,
            P_fp_use={k: 1 * v for k, v in P_fp_scn.items()},
            C_firm_use=C_firm_scn
        )

        m.Params.OutputFlag = 0
        m.optimize()

        if m.status != GRB.OPTIMAL:
            continue

        flows[(A0_val, L_val)] = {
            (f,p): m._N[(f,p)].X for f in F for p in P[f]
        }

    # ---- BASE CONFIG
    BASE = (1_000_000, 15)

    if BASE not in flows:
        print("Base missing")
        continue

    reference_flow = flows[BASE]
    baseline_active = {
    (f,p) for f in F for p in P[f]
    if reference_flow[(f,p)] > 1e-6}

    # ---- COMPARE
    for (A0_val, L_val), current_flow in flows.items():
        current_active = {
            (f,p) for f in F for p in P[f]
            if current_flow[(f,p)] > 1e-6
        }

        new_paths = current_active - baseline_active
        num_new_paths = len(new_paths)

        if (A0_val, L_val) == BASE:
            continue

        change = 0.0
        for f in F:
            for p in P[f]:
                change += abs(current_flow[(f,p)] - reference_flow[(f,p)])

        rerouted_pct = 100 * change / (2 * total_shipments)

       
        

        exp_results.append({
            "A0": A0_val,
            "lam": L_val,
            "new_paths": num_new_paths,
            "rerouted": rerouted_pct,
            "active_paths": sum(1 for f in F for p in P[f] if current_flow[(f,p)] > 1e-6),
            "instance": s
        })


df = pd.DataFrame(exp_results)
df.to_csv(OUTPUT_DIR / "new_path_results.csv", index=False)


df["rerouted_log"] = np.log1p(df["rerouted"])
print("\nResults:")
print(df)


from statsmodels.formula.api import ols

model = ols(
    'new_paths ~ C(A0) + C(lam) + C(instance) + C(A0):C(lam)',
    data=df
).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
anova_table.to_csv(OUTPUT_DIR / "new_path_anova.csv")

print("\nANOVA Table (New Paths):")
print(anova_table)
print("\nTotal runs:", len(df))
print("\nGroup means:")
print(df.groupby(["A0","lam"])["rerouted"].mean())
print("\nRerouting stats:")
print(df["rerouted"].describe())
print("\nNew paths (mean ± std):")
g = df.groupby(["A0","lam"])["new_paths"].agg(['mean','std'])
print(g)
# -------------------------
# COMPARISON BASELINE
# risk constraints ON + insurance OFF
# -------------------------




# -------------------------
# RESULTS
# -------------------------
if m.status == GRB.OPTIMAL:
    
    print("\n[Optimal]")
    print(f"Objective: {m.objVal:,.2f}")

    ops, prem, ship, total = objective_breakdown(m)
    print("\n[Objective breakdown]")
    print(f"  Operational costs      : {ops:,.2f}  ({100*ops/total:.2f}%)")
    print(f"  Expected premium cost : {prem:,.2f} ({100*prem/total:.2f}%)")
    print(f"  Expected uninsured loss   : {ship:,.2f} ({100*ship/total:.2f}%)")
    print(f"  Check total            : {total:,.2f} vs obj {m.objVal:,.2f}")

    # System risk usage
    sys_used = sum(R_fp[(f, p)] * m._N[(f, p)].X for f in F for p in P[f])
    print(f"\nSystem risk used: {sys_used:.6f} / {HARD_ALPHA:.6f} ({sys_used/HARD_ALPHA:.1%})")

    # Corridor usage
    print("\nCorridor usage (capped):")
    for k in K_cap:
        k_used = sum(R_fpk.get((f, p, k), 0.0) * m._N[(f, p)].X for f in F for p in P[f])
        cap = beta_k[k]
        print(f"  {k}: {k_used:.6f} / {cap:.6f} ({(k_used/cap if cap>0 else 0.0):.1%})")

    # If OTHER exists but not capped, still report it
    if (OTHER_ID in K_all) and (OTHER_ID not in K_cap):
        other_used = sum(R_fpk.get((f, p, OTHER_ID), 0.0) * m._N[(f, p)].X for f in F for p in P[f])
        print(f"\nOTHER ({OTHER_ID}) used (not capped): {other_used:.6f}")

    # Strategy summary
    print("\n[Strategy]")
    for f in F:
        used = [(p, m._N[(f, p)].X) for p in P[f]
                if m._X[(f, p)].X > 0.5 and m._N[(f, p)].X > 1e-6]
        used.sort(key=lambda x: -x[1])
        if used:
            show = ", ".join([f"{p}:{n:.1f}" for p, n in used[:6]])
            print(f"{f}: {show}")

elif m.status == GRB.INFEASIBLE:
    print("\nINFEASIBLE.")
    print("Common causes: PHI too tight, too few paths per facility, or segment bounds inconsistent.")
else:
    print("\nSolver status:", m.status)
print("change:", change)
print("2 * total_shipments:", 2 * total_shipments)
print(change <= 2 * total_shipments)
rerouted = change / (2 * total_shipments)
print("Percent rerouted:", 100 * rerouted)


for f,p in new_paths:
    print(f"Facility {f} → Path {p}")
