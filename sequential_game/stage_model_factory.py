"""Explicit, checked construction of the game's solver or learned stage transition.

The provisional surrogate candidate is GNN + prior2 + mixed game/SPE10 training.
The checkpoint is external to the repository and is supplied explicitly. The solver
remains the default/reference; use the same constructed model in both game roles to
run an all-surrogate world, or split realized and planning models for solver-grounded
evaluation of surrogate-based decisions.
"""
from pathlib import Path

import numpy as np


DEFAULT_CANDIDATE = {"backbone": "gnn", "hybrid": "prior2", "training": "mixed"}


class CheckedStageModel:
    """Validate the game/model boundary without changing transition or reward values."""

    def __init__(self, implementation, name, metadata=None):
        self.implementation = implementation
        self.name = name
        self.metadata = metadata or {}
        self.stage_time = float(implementation.stage_time)

    def initial_state(self, n, Swc=0.2):
        state = self.implementation.initial_state(n, Swc=Swc)
        self._check_state(state, (n, n), "initial_state")
        return state

    @staticmethod
    def _check_state(state, shape, where):
        if not isinstance(state, dict) or "p" not in state or "Sw" not in state:
            raise TypeError(f"{where} must provide a dict with p and Sw arrays")
        for key in ("p", "Sw"):
            arr = np.asarray(state[key])
            if arr.shape != shape:
                raise ValueError(f"{where}.{key} has shape {arr.shape}; expected {shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{where}.{key} contains non-finite values")

    def advance(self, geo, state, wells):
        if "k" not in geo or "phi" not in geo:
            raise KeyError("geology must contain k and phi")
        k, phi = np.asarray(geo["k"]), np.asarray(geo["phi"])
        if k.ndim != 2 or k.shape[0] != k.shape[1] or phi.shape != k.shape:
            raise ValueError(f"expected square, matching k/phi arrays; got {k.shape} and {phi.shape}")
        self._check_state(state, k.shape, "state")
        for well in wells:
            if len(well) != 3:
                raise ValueError(f"each well must be (i, j, rate), got {well!r}")
            i, j, rate = well
            if not (0 <= int(i) < k.shape[0] and 0 <= int(j) < k.shape[1]):
                raise IndexError(f"well coordinate {(i, j)} is outside grid {k.shape}")
            if not np.isfinite(rate) or rate < 0:
                raise ValueError(f"well rate must be finite and nonnegative, got {rate}")
        nxt, elapsed = self.implementation.advance(geo, state, wells)
        self._check_state(nxt, k.shape, "next_state")
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"transition elapsed time must be positive and finite, got {elapsed}")
        return nxt, elapsed


def create_stage_model(kind="solver", stage_time=1.0, checkpoint=None, device="cpu"):
    """Return `(checked_model, metadata)` for `solver` or `surrogate`.

    Surrogate checkpoints must use the same stage duration as the game. All physical
    arrays cross the boundary in solver units; StageSurrogate performs its documented
    float32 batching and network input normalization internally.
    """
    stage_time = float(stage_time)
    if kind == "solver":
        from stage_model import SolverStageModel

        impl = SolverStageModel(stage_time=stage_time)
        info = {"backend": "solver", "stage_time": stage_time, "checkpoint": None}
        return CheckedStageModel(impl, "solver", info), info
    if kind != "surrogate":
        raise ValueError("kind must be 'solver' or 'surrogate'")
    if not checkpoint:
        raise ValueError("surrogate backend requires --checkpoint pointing to a trained .pt file")
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"surrogate checkpoint not found: {path}")

    import torch
    from stage_surrogate import StageNet, StageSurrogate

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
    blob = torch.load(path, map_location=device)
    trained_time = float(blob.get("stage_time", 1.0))
    if not np.isclose(trained_time, stage_time, rtol=0.0, atol=1e-5):
        raise ValueError(f"checkpoint stage_time={trained_time:g} does not match game stage_time={stage_time:g}")
    net = StageNet.from_blob(blob)
    net.load_state_dict(blob["state_dict"])
    impl = StageSurrogate(net=net, stage_time=trained_time, device=device)
    args = blob.get("args", {})
    train_files = args.get("data", []) if isinstance(args, dict) else []
    info = {"backend": "surrogate", "checkpoint": str(path), "stage_time": trained_time,
            "backbone": net.backbone, "hybrid": net.hybrid, "n_sub": net.n_sub,
            "training_data_args": train_files,
            "candidate_architecture_match": net.backbone == DEFAULT_CANDIDATE["backbone"],
            "candidate_hybrid_match": net.hybrid == DEFAULT_CANDIDATE["hybrid"],
            "candidate_training_spec": DEFAULT_CANDIDATE["training"]}
    return CheckedStageModel(impl, "surrogate", info), info
