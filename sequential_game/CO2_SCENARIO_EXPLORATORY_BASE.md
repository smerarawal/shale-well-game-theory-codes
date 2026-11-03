# CO₂ surrogate-game scenario: exploratory base and sensitivities

Use `co2_surrogate_scenario_exploratory.json` for a reproducible six-stage
policy experiment on the current 24×24 CO₂ dataset. It is a normalized
research scenario, not a site-specific operating design, a permit limit, or
a dollar-valued cost model.

## Values and rationale

| Input | Base choice | Rationale and boundary |
|---|---:|---|
| Horizon and stage | 6 × 30 days | Matches `dataset_config.json`: six stages, 2,592,000 seconds per stage. This is a study clock, not a calibrated project schedule. |
| Discount factor | 1.0 per stage | No project discount rate is supplied. Use no discounting in the main six-stage screen to avoid inventing one; test annual effective rates separately and convert them to stage factors. |
| Maximum action rate | 1e−7 m³/s per operator's one fixed well | Matches the data-generation ceiling. A 30-day stage at 650 kg/m³ is 168.48 kg per operator at that rate. Keep planning actions within this training envelope. The planner takes this as `--rate-max-m3-s`; the top-level JSON field is provenance, not read by its CLI. |
| Pipeline capacity | 252.72 kg/stage | 75% of the combined maximum requests, 2 × 168.48 = 336.96 kg/stage. This creates controlled scarcity relative to the experiment's action grid; it is not a measured pipe capacity. `source_supply_kg_per_stage: null` leaves source supply nonbinding so the pilot isolates the pipe cap. |
| Storage cap | None in the base | Isolates the per-stage pipeline bottleneck. Add a cumulative storage-permit sensitivity separately rather than confounding two constraints in the baseline. |
| Allocation rule | Proportional | Transparent baseline matching the prior shared-capacity prototype. Compare CEA, CEL, and fixed-equal as separate policy arms. |
| Storage value | 1 normalized utility/kg | A declared normalization that rewards retained injection; it is not a market price, tax credit, or project revenue estimate. All payoffs must be labeled utility units. |
| Pressure limit | 5 MPa overpressure threshold | A stress-test threshold selected near the empirical upper tail of this model's pressure outputs. It is not a safe operating pressure or a fracture-pressure estimate. The solver initializes pressure at zero, so this is interpreted as simulated pressure rise. |
| Pressure damage | 1e−13 utility/Pa² | A transparent normalized penalty: a 1 MPa exceedance costs 0.1 utility units at that stage. Sweep the coefficient because it is not calibrated from money or geomechanics. |
| Costs and tariff | 0 in the base | Isolates the physical-capacity and pressure-incentive mechanisms. Add real resource costs and transfers only as separately labeled sensitivity arms. A tariff is a transfer unless its recipient and use enter the modeled welfare ledger. |
| Wells | (22,8), (22,16) | Matches the generator's two fixed wells for a 24×24 array: `z = nz−2`, `x = floor(nx/3)` and `floor(2nx/3)`. Coordinates are array indices `(z,x)`, not lengths. |
| Grid spacing | 50 ft × 10 ft × 4.1667 ft | Synthetic-section spacing derived from the declared 1,200 ft × 100 ft domain resampled to 24×24, with 10 ft width. For Tarbert/Ness, read the row's `dx_ft`, `dy_ft`, and `dz_ft` from its NPZ; vertical extent differs by section. |

Fluid properties and constitutive values in the JSON explicitly lock the
current solver defaults and dataset-generation settings: brine/CO₂ viscosity,
density, total compressibility, gravity, Corey endpoints/exponents, solubility,
dissolution rate, Land coefficient, CFL, and one-day maximum timestep. They
ensure reproducibility with the generated data; they are not independently
validated or field calibrated.

## Pressure-data audit that must accompany planning results

The attached `stage_metrics.csv` contains 7,488 transitions. Its row-level
pressure summary is: median pmax 0.694 MPa, 95th percentile 7.762 MPa, and
maximum 86.349 MPa; maximum stage-to-stage pmax increase is 21.856 MPa. The
largest absolute component mass-balance residual is 7.56e−12 kg. Mass closure
is strong numerically, but the high pressure tail still needs investigation.
Because injection is a point source on a coarse grid and the model has no
wellbore radius/near-well model, a very high cell pmax may be sensitive to grid
and source treatment. Report pmax, well-cell pressure, pressure outside a
near-well exclusion zone, and a grid/timestep sensitivity before treating the
5 MPa threshold as more than an exploratory model threshold.

For real CO₂ injection permits, a pressure limit must be tied to site-specific
fracture-pressure and confining-zone characterization. EPA Class VI guidance
requires injection pressure below 90% of injection-zone fracture pressure,
except during approved stimulation; the present simulator does not calculate
fracture pressure or full wellbore injection pressure. See [40 CFR 146.88(a)](https://www.law.cornell.edu/cfr/text/40/146.88) and [EPA Class VI guidance](https://www.epa.gov/sites/default/files/2015-07/documents/epa816r11020.pdf).

## Sensitivity arms

Run these as paired changes on the same geology/action seeds. These values are
chosen to probe the model envelope, not claimed as field distributions.

| Quantity | Sensitivity value 1 | Sensitivity value 2 | Sensitivity value 3 |
|---|---:|---:|---:|
| Max rate per well | 5e−8 m³/s | 7.5e−8 m³/s | 1e−7 m³/s |
| Pipeline capacity | 168.48 kg/stage | 252.72 kg/stage | 421.20 kg/stage |
| Pressure threshold | 2 MPa | 5 MPa | 10 MPa |
| Pressure damage coefficient | 1e−14 | 1e−13 | 1e−12 utility/Pa² |
| Storage value | 0.5 | 1.0 | 2.0 utility/kg |
| Real resource cost | 0 | 0.25 | 0.5 utility/kg |

The JSON base uses the data ceiling of `1e−7 m³/s` and no discounting. Also
run lower rate ceilings (`5e−8` and `7.5e−8 m³/s`). For discount sensitivity,
compare the base factor `1.0000` with `0.9960` (5% annual effective rate)
and `0.9922` (10% annual effective rate), converted over 30 days.

For storage permit sensitivity, the dataset's unconstrained action envelope is
2,021.76 kg over six stages. Try total permits of 1,010.88, 1,516.32, and
2,021.76 kg, plus no permit. To vary capacity when changing the rate ceiling,
derive the cap as a fraction of `2 × rate_max × stage_duration × rho_CO2`.
For common-source scarcity, vary source supply separately while keeping the
pipeline capacity fixed so their effects are identifiable.

## Interpretation boundary

Do not call these payoff units dollars, the 5 MPa pressure threshold a safety
limit, or the capacity values pipeline engineering estimates. The 2D model
supports controlled comparisons of solver-generated policies under stated
assumptions. A field-safe pressure limit requires fracture/stress data and a
calibrated well/reservoir pressure model; routed pipeline choices require a
network hydraulics and linepack model; 3D plume and completion effects require
a 3D reservoir solver.
