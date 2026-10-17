"""
spe10_geologies.py -- 12x12 geologies cut from SPE10 model 2 (Tarbert / Upper Ness layers),
rescaled to the solver's input range, for OUT-OF-DISTRIBUTION testing of the stage surrogate.

SPE10 model 2: 60 x 220 x 85 cells (20 x 10 x 2 ft). Layers 1-35 = Tarbert (smooth, near-shore),
layers 36-85 = Upper Ness (fluvial, channelised). Files in por_perm_case2.zip:
    spe_phi.dat   : 1,122,000 porosity values
    spe_perm.dat  : 3 x 1,122,000 values = all Kx, then all Ky, then all Kz (mD)
ASSUMED ordering (NOT verified here, no access to the real file): x varies fastest, then y,
then z, i.e. reshape(85, 220, 60) -> [z, y, x]. Check with --inspect on the real data: Upper Ness
layers should show channels, and the log-permeability should be spatially continuous.

Rescaling (so SPE10 stays inside the permeability/porosity range the solver and surrogate were
built for; solver dt shrinks as 1/max(k), and raw SPE10 spans many orders of magnitude):
    k   = 0.02 + 0.18 * minmax(log Kx over the patch)     -> [0.02, 0.2], same range as data_gen_v3
    phi = 0.08 + 0.17 * minmax(phi over the patch)        -> [0.08, 0.25]  (floored at 0.08)
Min-max is per patch, exactly like random_permeability/random_porosity. This keeps the spatial
STRUCTURE (channels) but compresses the contrast to the training range: it tests geometry,
not extreme heterogeneity. State this when reporting.

Patch choice changes the statistics: --coarsen f block-averages log k and phi by f x f before
cropping (f=1 = native cells, crops of 12 cells; f=4 = each model cell covers 4 SPE10 cells).
Cells are 20 x 10 ft (not square); this is ignored, as in the solver's dx = 1.

    python spe10_geologies.py --inspect --data_dir /kaggle/input/spe10

load_spe10 accepts either spe_perm.dat + spe_phi.dat, or SPE10MODEL2_PERM.INC + SPE10MODEL2_PHI.INC.
Eclipse ordering is x fastest, then y, then z with layer 1 on top (same assumption as above).
"""
import argparse
import os

import numpy as np

NX, NY, NZ = 60, 220, 85
N_CELLS = NX * NY * NZ
TARBERT = range(0, 35)
UPPER_NESS = range(35, 85)


def _read_flat(path):
    return np.fromstring(open(path).read(), sep=" ") if os.path.getsize(path) < 1 << 28 else np.loadtxt(path).ravel()


def read_eclipse_arrays(path):
    """Parse an Eclipse-style .INC file -> {KEYWORD: flat float array}.
    Handles '--' comments, '/' terminators, 'n*value' repeats, and keyword lines (PERMX, PORO, ...).
    A file with no keyword at all is returned under the key ''. The exact SPE10 .INC layout was NOT
    checked against the real files (site blocks automated access), so the keyword names found are
    printed by the caller; treat a size mismatch as a sign the parse is wrong."""
    out, cur, vals = {}, "", []

    def flush():
        if vals:
            out.setdefault(cur, []).extend(vals)

    for line in open(path):
        line = line.split("--")[0].strip()
        if not line:
            continue
        for tok in line.replace("/", " / ").split():
            if tok == "/":
                flush(); vals = []; cur = ""
            elif tok[0].isalpha():
                flush(); vals = []; cur = tok.upper()
            elif "*" in tok:
                n, v = tok.split("*")
                vals.extend([float(v)] * int(n))
            else:
                vals.append(float(tok))
    flush()
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _find(data_dir, *names):
    """Case-insensitive file lookup (Kaggle/zip tools sometimes change the case)."""
    have = {f.lower(): os.path.join(data_dir, f) for f in os.listdir(data_dir)}
    for n in names:
        if n.lower() in have:
            return have[n.lower()]
    return None


def load_spe10(data_dir, phi_floor=0.01):
    """Returns logK [NZ,NY,NX] (Kx, log10 mD) and phi [NZ,NY,NX].
    Accepts EITHER spe_perm.dat + spe_phi.dat (SPE web page zip) OR the Eclipse deck files
    SPE10MODEL2_PERM.INC + SPE10MODEL2_PHI.INC (SPE10_MODEL2 folder of the Eclipse data set).
    Porosity is floored at phi_floor: the Eclipse deck replaces zero porosities by 1e-7, which would
    otherwise dominate the per-patch min-max rescaling."""
    perm_f, phi_f = _find(data_dir, "spe_perm.dat"), _find(data_dir, "spe_phi.dat")
    if perm_f and phi_f:
        perm, phi = _read_flat(perm_f), _read_flat(phi_f)
        kx = perm[:N_CELLS]
        assert perm.size == 3 * N_CELLS, f"{perm_f} has {perm.size} values, expected {3 * N_CELLS}"
    else:
        perm_f = _find(data_dir, "SPE10MODEL2_PERM.INC")
        phi_f = _find(data_dir, "SPE10MODEL2_PHI.INC")
        assert perm_f and phi_f, f"no spe_perm.dat/spe_phi.dat and no SPE10MODEL2_PERM.INC/PHI.INC in {data_dir}"
        pa, fa = read_eclipse_arrays(perm_f), read_eclipse_arrays(phi_f)
        print("keywords found:", {k: v.size for k, v in pa.items()}, {k: v.size for k, v in fa.items()})
        kx = pa.get("PERMX", pa.get("")) if "PERMX" in pa or "" in pa else None
        assert kx is not None, f"no PERMX block in {perm_f}: found {list(pa)}"
        phi = fa.get("PORO", fa.get("PHI", fa.get("", next(iter(fa.values())))))
    assert kx.size == N_CELLS, f"Kx has {kx.size} values, expected {N_CELLS}"
    assert phi.size == N_CELLS, f"porosity has {phi.size} values, expected {N_CELLS}"
    return np.log10(np.maximum(kx.reshape(NZ, NY, NX), 1e-6)), np.maximum(phi.reshape(NZ, NY, NX), phi_floor)


def _minmax(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _coarsen(a, f):
    if f == 1:
        return a
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def sample_patches(logk, phi, layers, n, n_cells=12, coarsen=1, seed=0, min_contrast=0.3):
    """n random n_cells x n_cells patches from random layers in `layers`.
    Patches whose log10 K range is below `min_contrast` decades are rejected (near-uniform
    patches carry no structure and would dominate after per-patch min-max rescaling)."""
    rng = np.random.default_rng(seed)
    layers = list(layers)
    out = []
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        z = int(rng.choice(layers))
        lk, ph = _coarsen(logk[z], coarsen), _coarsen(phi[z], coarsen)
        i = int(rng.integers(0, lk.shape[0] - n_cells + 1))
        j = int(rng.integers(0, lk.shape[1] - n_cells + 1))
        pk, pp = lk[i:i + n_cells, j:j + n_cells], ph[i:i + n_cells, j:j + n_cells]
        if pk.max() - pk.min() < min_contrast:
            continue
        out.append({"k": 0.02 + 0.18 * _minmax(pk), "phi": 0.08 + 0.17 * _minmax(pp), "layer": z,
                    "i": i, "j": j, "logk_range": float(pk.max() - pk.min())})
    if len(out) < n:
        raise RuntimeError(f"only {len(out)}/{n} patches passed the contrast filter")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing spe_perm.dat and spe_phi.dat")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    logk, phi = load_spe10(a.data_dir)
    print("log10 Kx range per region:")
    for nm, rg in (("Tarbert", TARBERT), ("Upper Ness", UPPER_NESS)):
        sl = logk[list(rg)]
        print(f"  {nm}: min {sl.min():.2f} median {np.median(sl):.2f} max {sl.max():.2f}; phi {phi[list(rg)].min():.3f}-{phi[list(rg)].max():.3f}")
    if a.inspect:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(9, 5))
        for a_, z, t in zip(ax, (10, 45, 75), ("Tarbert z=10", "Upper Ness z=45", "Upper Ness z=75")):
            a_.imshow(logk[z], aspect="auto"); a_.set_title(t)
        plt.savefig("spe10_inspect.png", dpi=110)
        print("wrote spe10_inspect.png -- Upper Ness should show channels; if it looks like noise the file order is wrong")


if __name__ == "__main__":
    main()
