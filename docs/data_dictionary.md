# Data dictionary

The optimization scripts read `data/master_data.xlsx`.

## Workbook sheets

| Sheet | Rows | Key fields | Purpose |
| --- | ---: | --- | --- |
| `facilities` | 40 | `facility_id`, `FACILITY NAME`, `D_f` | Facility identifier, name, and trip demand |
| `paths` | 400 | `facility_id`, `path_id`, `P_fp`, `R_fp`, `U_fp` | Candidate-path probability, risk, and activation cost |
| `corridors` | 8 | `Corridor Name`, `corr_id`, `beta_k` | Tracked corridor identifiers and stored caps |
| `region risk` | 6 | `region_id`, `beta_m` | Regional identifiers and stored limits |
| `corridor_path_risk` | 1,208 | `facility_id`, `path_id`, `corridor_id`, `R_fpk` | Raw corridor attribution by facility-path pair |
| `segments` | 10 | `s`, `L`, `U`, `c`, `F` | Flow bounds, unit cost, and stored segment charge |
| `region_path_risk` | 1,295 | `facility_id`, `path_id`, `region_id`, `R_fpm` | Regional risk attribution by facility-path pair |

## Important implementation notes

- Every facility has 10 candidate paths.
- Total demand is 506,613 trips.
- `P_fp` ranges from approximately \(1.07\times10^{-7}\) to \(9.93\times10^{-7}\) in the supplied workbook.
- Corridor contributions are overlap-based and do not initially sum to `R_fp`; each script normalizes them before building the model.
- Regional contributions already sum to `R_fp` within floating-point tolerance.
- The `F` column in `segments` is present in the workbook but is not used by the supplied objective.
- Two empty unnamed columns remain in the `corridor_path_risk` sheet and are ignored by the scripts.

## Other files

- `data/intermediate/` contains upstream path-risk and attribution tables.
- `data/results/` contains previously generated sensitivity outputs.
- Raw road-network, Census geometry, ArcGIS, and route-generation files are not included.

Facility names are retained in this private review release. Confirm data-source terms and advisor approval before making the repository public.
