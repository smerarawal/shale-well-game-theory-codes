"""CO2 study utilities: geology cohorts, diagnostics, and payoff accounting.

The geology regimes are controlled *game-interaction* regimes, not claims that
one rock is objectively easy/hard to store in. All fields are randomized within
their recipe. SPE10 crops remain a separate benchmark cohort.

The payoff function expects solver-derived CO2 mass/plume diagnostics. It does
not infer dissolution or trapping from pressure/saturation arrays; those must
come from a validated CO2 transition model.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

@dataclass(frozen=True)
class RegimeSpec:
    name: str
    layout: str
    corr_len: float
    barrier_strength: float = 1.0
    bridge_strength: float = 0.0


# Far separation plus a low-permeability band weakens cross-operator flow.
# Close separation plus a connected high-k bridge strengthens it. The middle
# regime retains the randomized field structure used in earlier experiments.
REGIMES = {
    "friendly": RegimeSpec("friendly", "far", corr_len=3.0, barrier_strength=0.08),
    "intermediate": RegimeSpec("intermediate", "mid", corr_len=6.0),
    "challenging": RegimeSpec("challenging", "close", corr_len=8.0, bridge_strength=0.85),
}


def _unit_random_field(shape, seed, corr_len):
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal(shape), sigma=corr_len, mode="reflect")
    lo, hi = np.quantile(field, [0.01, 0.99])
    return np.clip((field - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def make_regime_geology(seed, n=12, regime="intermediate", mult=None):
    """Generate one randomized synthetic field and its pre-game metadata.

    `friendly` adds a vertical low-k separator; `challenging` adds a randomized
    connected high-k bridge between the operator regions. These are design
    treatments intended to change interference. The measured interference
    diagnostic, not the label, determines whether each generated case achieved
    its intended regime.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {tuple(REGIMES)}")
    spec = REGIMES[regime]
    rng = np.random.default_rng(seed)
    # Convert smooth random fields to the same normalized range as the earlier
    # synthetic data generator; vary multiplier as before, but keep it bounded.
    k01 = _unit_random_field((n, n), int(seed), spec.corr_len)
    phi01 = _unit_random_field((n, n), int(seed) + 10_000, max(1.0, spec.corr_len * 0.7))
    k = 0.02 + 0.18 * k01
    phi = 0.08 + 0.17 * phi01

    center = n // 2 + int(rng.integers(-max(1, n // 12), max(2, n // 12 + 1)))
    if regime == "friendly":
        width = max(1, n // 12)
        # Operators are separated along x, so the weak-permeability barrier
        # must be a vertical band in the [z, x] array.
        k[:, max(0, center - width):min(n, center + width + 1)] *= spec.barrier_strength
    elif regime == "challenging":
        # A gently meandering, randomized band connects the two operator zones.
        width = max(1, n // 10)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude = max(1.0, n * 0.12)
        for j in range(n):
            i0 = int(np.clip(center + amplitude * np.sin(2 * np.pi * j / max(n, 1) + phase), 0, n - 1))
            k[max(0, i0 - width):min(n, i0 + width + 1), j] = np.maximum(
                k[max(0, i0 - width):min(n, i0 + width + 1), j],
                0.02 + 0.18 * spec.bridge_strength)

    if mult is None:
        mult = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    k = np.clip(k * float(mult), 0.005, 0.4)
    return {"k": k.astype(np.float64), "phi": phi.astype(np.float64),
            "seed": int(seed), "mult": float(mult), "cohort": regime,
            "layout": spec.layout, "corr_len": float(spec.corr_len),
            "generation": "co2_controlled_interference_v1"}


def generate_synthetic_cohorts(n_per_regime, seed=41000, n=12):
    """Return randomized fields in the friendly, intermediate and challenging regimes."""
    if n_per_regime < 1:
        raise ValueError("n_per_regime must be positive")
    cohorts = {}
    for ri, regime in enumerate(REGIMES):
        cohorts[regime] = [make_regime_geology(seed + ri * 100_000 + i, n=n, regime=regime)
                           for i in range(n_per_regime)]
    return cohorts


def game_config_for_geology(geo, **overrides):
    """Build the existing game configuration using the regime's well layout.

    This prepares the layout treatment only. CO2 payoff integration and a
    CO2-specific game runner are still separate implementation steps.
    """
    if "k" not in geo:
        raise ValueError("This geology is a directional x-z data section; the existing scalar-k 2D game cannot consume it yet")
    from game3 import Cfg

    values = {"n": int(np.asarray(geo["k"]).shape[0]),
              "layout": geo.get("layout", "mid")}
    values.update(overrides)
    return Cfg(**values)


def generate_spe10_cohort(data_dir, n_per_region=24, n=12, seed=81000,
                          coarsen=1, permeability_multipliers=(0.8, 1.0, 1.25)):
    """Sample Tarbert and Upper Ness patches as a distinct benchmark cohort.

    SPE10's original field arrays remain untouched by the synthetic generators.
    The existing loader maps each sampled patch into the solver's normalized
    permeability/porosity range; metadata records that this is a scaled patch.
    """
    from spe10_geologies import TARBERT, UPPER_NESS, load_spe10, sample_patches

    logk, phi = load_spe10(str(data_dir))
    result = {}
    for ri, (name, layers) in enumerate((("tarbert", TARBERT), ("upper_ness", UPPER_NESS))):
        patches = sample_patches(logk, phi, layers, n_per_region, n_cells=n,
                                 coarsen=coarsen, seed=seed + ri, band="all")
        cases = []
        for i, patch in enumerate(patches):
            mult = float(permeability_multipliers[i % len(permeability_multipliers)])
            cases.append({"k": np.clip(patch["k"] * mult, 0.005, 0.4).astype(np.float64),
                          "phi": patch["phi"].astype(np.float64),
                          "seed": int(seed + ri * 100_000 + i), "mult": mult,
                          "cohort": "spe10_" + name, "spe10_layer": int(patch["layer"]),
                          "spe10_i": int(patch["i"]), "spe10_j": int(patch["j"]),
                          "spe10_logk_range": float(patch["logk_range"]),
                          "generation": "spe10_scaled_patch_v1"})
        result[name] = cases
    return result


def generate_spe10_xz_cohort(data_dir, n_per_region=24, seed=82000,
                             coarsen_z=1, coarsen_x=1,
                             permeability_multipliers=(1.0,)):
    """Sample vertical SPE10 x-z sections while retaining raw directional K (mD).

    SPE10 arrays are stored as [z,y,x]. Each case fixes one y row, then takes the
    complete Tarbert or Upper Ness vertical band. The output fields are [z,x];
    ``kx_mD`` and ``kz_mD`` remain separate, so this data cannot be mistaken for
    the old square-grid scalar-k input. Coarsening uses geometric block means of
    permeability and arithmetic means of porosity as a data-reduction convention;
    it is not a transmissibility-preserving upscaling method.
    """
    from spe10_geologies import TARBERT, UPPER_NESS, load_spe10_components

    if coarsen_z < 1 or coarsen_x < 1:
        raise ValueError("coarsen_z and coarsen_x must be positive integers")
    # Preserve SPE10 porosity in this vertical data export; the legacy 2D
    # surrogate loader's phi floor is only for its min-max-scaled patch path.
    fields = load_spe10_components(str(data_dir), phi_floor=0.0)
    result = {}
    bands = (("tarbert", TARBERT), ("upper_ness", UPPER_NESS))
    if n_per_region > fields["phi"].shape[1]:
        raise ValueError("n_per_region cannot exceed the number of SPE10 y slices when sampling without replacement")
    # Use matched, unique y slices for Tarbert and Upper Ness. This prevents
    # duplicate sections and supports geology-level holdouts across both bands.
    y_indices = np.random.default_rng(seed).choice(fields["phi"].shape[1],
                                                    size=n_per_region, replace=False)
    for ri, (name, layers) in enumerate(bands):
        cases = []
        z0, z1 = min(layers), max(layers) + 1
        for i in range(n_per_region):
            y = int(y_indices[i])
            mult = float(permeability_multipliers[i % len(permeability_multipliers)])
            arrays = {}
            for key in ("kx_mD", "kz_mD"):
                section = fields[key][z0:z1, y, :]
                nz = (section.shape[0] // coarsen_z) * coarsen_z
                nx = (section.shape[1] // coarsen_x) * coarsen_x
                section = section[:nz, :nx]
                # Geometric means preserve orders of magnitude better than arithmetic means.
                blocks = np.log(np.maximum(section, 1e-12)).reshape(
                    nz // coarsen_z, coarsen_z, nx // coarsen_x, coarsen_x)
                arrays[key] = np.exp(blocks.mean(axis=(1, 3))) * mult
            section_phi = fields["phi"][z0:z1, y, :]
            nz = (section_phi.shape[0] // coarsen_z) * coarsen_z
            nx = (section_phi.shape[1] // coarsen_x) * coarsen_x
            arrays["phi"] = section_phi[:nz, :nx].reshape(
                nz // coarsen_z, coarsen_z, nx // coarsen_x, coarsen_x).mean(axis=(1, 3))
            cases.append({**arrays, "seed": int(seed + ri * 100_000 + i), "mult": mult,
                          "cohort": "spe10_xz_" + name, "representation": "vertical_xz_section",
                          "orientation": "array_axis_0=z_vertical; array_axis_1=x_horizontal",
                          "spe10_y_index": y, "spe10_z_start": z0,
                          "spe10_z_end_exclusive": z1,
                          "dx_ft": float(20 * coarsen_x), "dy_ft": 10.0,
                          "dz_ft": float(2 * coarsen_z),
                          "coarsen_x": int(coarsen_x), "coarsen_z": int(coarsen_z),
                          "permeability_units": "mD", "porosity_units": "fraction",
                          "generation": "spe10_raw_directional_xz_section_v1"})
        result[name] = cases
    return result


def geology_descriptors(geo, high_k_quantile=0.8):
    """Compute reproducible field-only descriptors before game outcomes."""
    # For directional SPE10 sections, use horizontal Kx for the connectivity
    # descriptor and keep Kz separately in the saved source arrays.
    k_key = "k" if "k" in geo else "kx_mD" if "kx_mD" in geo else None
    if k_key is None:
        raise ValueError("geology must contain scalar k or directional kx_mD")
    k, phi = np.asarray(geo[k_key], float), np.asarray(geo["phi"], float)
    if k.ndim != 2 or phi.shape != k.shape:
        raise ValueError("geology k and phi must be matching 2D arrays")
    mask = k >= np.quantile(k, high_k_quantile)
    # Four-neighbor connected fraction gives a simple channel-connectivity
    # proxy without claiming to be a full geological facies interpretation.
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for start in zip(*np.where(mask & ~seen)):
        stack, size = [start], 0
        seen[start] = True
        while stack:
            i, j = stack.pop(); size += 1
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] and not seen[ni, nj]:
                    seen[ni, nj] = True; stack.append((ni, nj))
        components.append(size)
    return {"logk_std": float(np.std(np.log(np.maximum(k, 1e-12)))),
            "phi_mean": float(phi.mean()), "phi_std": float(phi.std()),
            "high_k_fraction": float(mask.mean()),
            "largest_high_k_component_fraction": float(max(components, default=0) / mask.size),
            "high_k_component_count": int(len(components))}


@dataclass(frozen=True)
class CO2PayoffConfig:
    """Coefficients are payoff units per solver unit until calibrated."""
    injection_margin: float = 20.0
    stored_mass_credit: float = 1.0
    drilling_cost: float = 18.0
    pressure_cap: float = 6.0
    pressure_damage: float = 30.0
    plume_overlap_damage: float = 1.0
    boundary_contact_damage: float = 1.0


def score_co2_stage(*, rates_by_operator, elapsed, new_wells_by_operator,
                    pmax, retained_mass_by_operator, plume_overlap,
                    boundary_contact_fraction, cfg=CO2PayoffConfig()):
    """Score a CO2 stage with revenue, retained-storage value and both harms.

    `retained_mass_by_operator` must come from the CO2 component/tracer ledger;
    plume overlap and boundary contact are fractions in [0,1]. Pressure damage
    is a shared field cost and is split by injection volume, matching the
    baseline game's pro-rata liability convention.
    """
    rates = np.asarray(rates_by_operator, dtype=float)
    retained = np.asarray(retained_mass_by_operator, dtype=float)
    drilled = np.asarray(new_wells_by_operator, dtype=float)
    if rates.ndim != 1 or retained.shape != rates.shape or drilled.shape != rates.shape:
        raise ValueError("operator rate, retained-mass, and drilling vectors must have the same shape")
    if elapsed <= 0 or np.any(rates < 0) or np.any(retained < 0) or np.any(drilled < 0):
        raise ValueError("elapsed must be positive and operator quantities nonnegative")
    overlap = float(np.clip(plume_overlap, 0.0, 1.0))
    boundary = float(np.clip(boundary_contact_fraction, 0.0, 1.0))
    injected = rates * float(elapsed)
    pressure_cost = cfg.pressure_damage * max(0.0, float(pmax) - cfg.pressure_cap) ** 2
    overlap_cost = cfg.plume_overlap_damage * overlap
    boundary_cost = cfg.boundary_contact_damage * boundary
    harm = pressure_cost + overlap_cost + boundary_cost
    shares = injected / injected.sum() if injected.sum() > 1e-12 else np.zeros_like(injected)
    revenue = cfg.injection_margin * injected
    storage_credit = cfg.stored_mass_credit * retained
    drilling = cfg.drilling_cost * drilled
    liability = harm * shares
    payoff = revenue + storage_credit - drilling - liability
    return {"payoff": payoff, "revenue": revenue, "storage_credit": storage_credit,
            "drilling_cost": drilling, "pressure_cost": pressure_cost,
            "plume_overlap_cost": overlap_cost, "boundary_contact_cost": boundary_cost,
            "liability_share": shares, "total_welfare": float(payoff.sum()),
            "injected_mass_proxy": injected, "retained_mass": retained}


def save_geology_manifest(cohorts, path):
    """Write a compact CSV manifest for reproducible cohort sampling."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("cohort", "seed", "mult", "layout", "corr_len", "representation",
              "orientation", "dx_ft", "dy_ft", "dz_ft", "coarsen_x", "coarsen_z",
              "permeability_units", "porosity_units", "spe10_y_index",
              "spe10_z_start", "spe10_z_end_exclusive", "spe10_layer",
              "spe10_i", "spe10_j", "spe10_logk_range", "logk_std", "phi_mean",
              "phi_std", "high_k_fraction", "largest_high_k_component_fraction",
              "high_k_component_count", "generation")
    rows = []
    for group in cohorts.values():
        for geo in group:
            rows.append({**geo, **geology_descriptors(geo)})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return path


def save_cohort_arrays(cohorts, path):
    """Save fixed-shape geology arrays and primitive metadata to compressed NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    field_keys = ("k", "kx_mD", "ky_mD", "kz_mD", "phi")
    for cohort_name, cases in cohorts.items():
        if not cases:
            continue
        present = [key for key in field_keys if all(key in g for g in cases)]
        for key in present:
            arrays[f"{cohort_name}_{key}"] = np.stack([g[key] for g in cases]).astype(np.float32)
        arrays[f"{cohort_name}_seed"] = np.asarray([g["seed"] for g in cases], dtype=np.int64)
        arrays[f"{cohort_name}_mult"] = np.asarray([g["mult"] for g in cases], dtype=np.float32)
        arrays[f"{cohort_name}_metadata_json"] = np.asarray([
            {k: v for k, v in g.items() if k not in field_keys} for g in cases], dtype=object)
    np.savez_compressed(path, **arrays)
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate CO2 geology cohorts and a manifest.")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n", type=int, default=12, help="square grid width")
    parser.add_argument("--n-per-regime", type=int, default=24)
    parser.add_argument("--spe10-dir", default=None, help="optional folder with SPE10 .INC or .dat files")
    parser.add_argument("--n-per-spe10-region", type=int, default=24)
    parser.add_argument("--spe10-xz", action="store_true",
                        help="also save vertical x-z SPE10 sections with raw directional Kx/Kz")
    parser.add_argument("--coarsen-z", type=int, default=1)
    parser.add_argument("--coarsen-x", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--coarsen", type=int, default=1)
    args = parser.parse_args()

    cohorts = generate_synthetic_cohorts(args.n_per_regime, seed=args.seed, n=args.n)
    if args.spe10_dir:
        spe = generate_spe10_cohort(args.spe10_dir, n_per_region=args.n_per_spe10_region,
                                    n=args.n, seed=args.seed + 1_000_000, coarsen=args.coarsen)
        cohorts.update({f"spe10_{name}": cases for name, cases in spe.items()})
        if args.spe10_xz:
            xz = generate_spe10_xz_cohort(
                args.spe10_dir, n_per_region=args.n_per_spe10_region,
                seed=args.seed + 2_000_000, coarsen_z=args.coarsen_z,
                coarsen_x=args.coarsen_x)
            cohorts.update({f"spe10_xz_{name}": cases for name, cases in xz.items()})
    out = Path(args.out)
    save_cohort_arrays(cohorts, out / "co2_geology_cohorts.npz")
    save_geology_manifest(cohorts, out / "co2_geology_manifest.csv")
    print(f"Wrote {sum(map(len, cohorts.values()))} geology fields to {out}")


if __name__ == "__main__":
    main()
