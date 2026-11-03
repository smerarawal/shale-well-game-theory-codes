"""Differentiable finite-horizon best responses through a CO2 surrogate.

This is a planning/label-generation tool, not a replacement for the solver.
It optimizes open-loop stage rates for two operators through a trained x-z
surrogate, then emits actions that must be re-evaluated with the CO2 solver.
The smooth pressure penalty is an optimization aid; reported payoffs should be
computed with the nonsmoothed game definition and solver states.

Example:
  python co2_differentiable_game.py --data-dir DATA --checkpoint MODEL.pt \\
      --cohort challenging --geology-seed 41000 --out best_response.json \\
      --rate-max-m3-s 1e-7 --pressure-cap-pa 2000000 --pressure-damage-per-pa2 1e-10
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import train_co2_xz_surrogate as trainer
except ImportError:
    import train_co2_xz_surrogate as trainer


def _torch_dispatch(mass: torch.Tensor, capacity: torch.Tensor,
                    rule: str) -> torch.Tensor:
    """Differentiable piecewise approximation of game pipeline dispatch."""
    total = mass.sum()
    if rule == "proportional":
        return mass * torch.clamp(capacity / total.clamp_min(1e-12), max=1.0)
    if rule == "fixed_equal":
        return torch.minimum(mass, capacity / mass.numel())
    lo = torch.zeros((), device=mass.device, dtype=mass.dtype)
    hi = mass.max()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if rule == "cea":
            award = torch.minimum(mass, mid)
            below_capacity = award.sum() < capacity
            lo = torch.where(below_capacity, mid, lo)
            hi = torch.where(below_capacity, hi, mid)
        elif rule == "cel":
            award = torch.clamp(mass - mid, min=0.0)
            above_capacity = award.sum() > capacity
            lo = torch.where(above_capacity, mid, lo)
            hi = torch.where(above_capacity, hi, mid)
        else:
            raise ValueError("dispatch rule must be proportional, fixed_equal, cea, or cel")
    cut = 0.5 * (lo + hi)
    return torch.minimum(mass, cut) if rule == "cea" else torch.clamp(mass - cut, min=0.0)


def _state_at(data: dict[str, np.ndarray], geology_seed: int, episode_id: int | None):
    mask = (data["geology_seed"] == geology_seed) & (data["stage_index"] == 0)
    if episode_id is not None:
        mask &= data["episode_id"] == episode_id
    ids = np.flatnonzero(mask)
    if not len(ids):
        raise ValueError(f"no stage-zero sample for geology_seed={geology_seed}, episode={episode_id}")
    return int(ids[0])


def _features(data: dict[str, np.ndarray], row: int, dynamic: torch.Tensor,
              rates: torch.Tensor, rate_scale: float) -> torch.Tensor:
    """Build the trainer's 14-channel input while preserving rate gradients."""
    device, dtype = dynamic.device, dynamic.dtype
    _, h, w = data["kx_mD"].shape
    t = lambda key: torch.as_tensor(data[key][row:row + 1], device=device, dtype=dtype)
    logkx = torch.log(t("kx_mD").clamp_min(1e-12))
    logkz = torch.log(t("kz_mD").clamp_min(1e-12))
    phi = t("phi")
    coords_x = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, w).expand(1, h, w)
    coords_z = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, h, 1).expand(1, h, w)
    rate_maps = [torch.zeros((1, h, w), device=device, dtype=dtype) for _ in range(2)]
    wz = h - 2
    wx = (max(1, w // 3), min(w - 2, (2 * w) // 3))
    # CopySlices retains gradients from rates into these single-cell maps.
    rate_maps[0][0, wz, wx[0]] = rates[0] / rate_scale
    rate_maps[1][0, wz, wx[1]] = rates[1] / rate_scale
    logdx = torch.log(t("dx_ft").clamp_min(1e-6)).view(1, 1, 1).expand(1, h, w)
    logdz = torch.log(t("dz_ft").clamp_min(1e-6)).view(1, 1, 1).expand(1, h, w)
    return torch.stack((logkx, logkz, phi, dynamic[:, 0], dynamic[:, 1],
                        dynamic[:, 2], dynamic[:, 3], dynamic[:, 4] / 30.0,
                        rate_maps[0], rate_maps[1], coords_x, coords_z,
                        logdx, logdz), dim=1)


def _one_player_objective(
    model: torch.nn.Module, data: dict[str, np.ndarray], row: int,
    initial: torch.Tensor, rates: torch.Tensor, stats: dict[str, np.ndarray],
    rate_scale: float, *, player: int, opponent_rates: torch.Tensor,
    max_rate_m3_s: float, stage_days: float, rho: float,
    storage_value_per_kg: float, pipeline_cost_per_kg: float,
    pipeline_tariff_per_kg: float, storage_cap_kg: float | None,
    pipeline_dispatch_rule: str, storage_dispatch_rule: str,
    pressure_cap_pa: float, pressure_damage_per_pa2: float,
    pressure_smoothing_pa: float, pipeline_capacity_kg_per_stage: float | None,
    discount_factor: float,
) -> torch.Tensor:
    """Differentiably roll six stages and return one operator's modeled payoff."""
    device = initial.device
    dt = stage_days * 86400.0
    y_mean = torch.as_tensor(stats["y_mean"], device=device)
    y_std = torch.as_tensor(stats["y_std"], device=device)
    x_mean = torch.as_tensor(stats["x_mean"], device=device)
    x_std = torch.as_tensor(stats["x_std"], device=device)
    dynamic = initial
    hidden = None
    injected_by_stage = []
    cumulative_injected = torch.zeros((), device=device)
    cumulative_by_operator = torch.zeros(2, device=device)
    discounted_cost = torch.zeros((), device=device)
    horizon = rates.shape[0]
    for stage in range(horizon):
        requested = torch.stack((rates[stage], opponent_rates[stage])) if player == 0 else torch.stack((opponent_rates[stage], rates[stage]))
        masses = requested * (rho * dt)
        if pipeline_capacity_kg_per_stage is not None:
            masses = _torch_dispatch(
                masses, torch.as_tensor(pipeline_capacity_kg_per_stage, device=device),
                pipeline_dispatch_rule)
        if storage_cap_kg is not None:
            cap = torch.as_tensor(storage_cap_kg, device=device)
            if storage_dispatch_rule == "fixed_equal":
                quotas = torch.clamp(cap / 2.0 - cumulative_by_operator, min=0.0)
                masses = torch.minimum(masses, quotas)
            else:
                remaining = torch.clamp(cap - cumulative_injected, min=0.0)
                masses = _torch_dispatch(masses, remaining, storage_dispatch_rule)
        cumulative_injected = cumulative_injected + masses.sum()
        cumulative_by_operator = cumulative_by_operator + masses
        delivered_rates = masses / (rho * dt)
        actual_rates = delivered_rates[player]
        injected_by_stage.append(masses[player])
        x = _features(data, row, dynamic, delivered_rates, rate_scale)
        x = (x - x_mean) / x_std
        if hasattr(model, "step"):
            pred_norm, hidden = model.step(x, hidden)
        else:
            pred_norm = model(x)
        raw_next = dynamic + pred_norm * y_std + y_mean
        dynamic = trainer._torch_project(raw_next, dynamic)
        pmax = dynamic[:, 0].flatten(1).amax(1)[0]
        # Softplus approximates max(0, pmax - cap) during gradient search.
        smooth_exceed = F.softplus((pmax - pressure_cap_pa) /
                                   max(pressure_smoothing_pa, 1.0)) * max(pressure_smoothing_pa, 1.0)
        shared_damage = pressure_damage_per_pa2 * smooth_exceed.square()
        share = masses[player] / masses.sum().clamp_min(1e-12)
        cost = ((pipeline_cost_per_kg + pipeline_tariff_per_kg) * masses[player]
                + shared_damage * share)
        discounted_cost = discounted_cost + (discount_factor ** stage) * cost
    # The current closed-boundary game credits cumulative injected mass at the
    # terminal stage. Storage credit is therefore terminal, not double counted.
    terminal_credit = storage_value_per_kg * torch.stack(injected_by_stage).sum()
    terminal_credit = (discount_factor ** (horizon - 1)) * terminal_credit
    return terminal_credit - discounted_cost


def iterated_best_responses(
    model: torch.nn.Module, data: dict[str, np.ndarray], row: int,
    stats: dict[str, np.ndarray], rate_scale: float, *, horizon: int = 6,
    rate_max_m3_s: float, stage_days: float = 30.0, rho: float = 650.0,
    storage_value_per_kg: float = 1.0, pipeline_cost_per_kg: float = 0.0,
    pipeline_tariff_per_kg: float = 0.0, storage_cap_kg: float | None = None,
    pipeline_dispatch_rule: str = "proportional",
    storage_dispatch_rule: str = "proportional",
    pressure_cap_pa: float, pressure_damage_per_pa2: float,
    pressure_smoothing_pa: float = 5e4,
    pipeline_capacity_kg_per_stage: float | None = None,
    discount_factor: float = 1 / 1.05, rounds: int = 4,
    steps_per_response: int = 100, learning_rate: float = 0.08,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Alternate differentiable open-loop best responses between two operators."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    initial_np = np.stack([data[f"{key}_in"][row] for key in trainer.STATE_KEYS], axis=0)[None]
    initial = torch.as_tensor(initial_np, dtype=torch.float32, device=dev)
    # Sigmoid parameterization keeps requested rates within [0, rate_max].
    logits = [torch.zeros(horizon, dtype=torch.float32, device=dev, requires_grad=True)
              for _ in range(2)]
    optimizers = [torch.optim.Adam([x], lr=learning_rate) for x in logits]
    objectives: list[list[float]] = [[], []]
    for _round in range(rounds):
        for player in range(2):
            for _ in range(steps_per_response):
                optimizers[player].zero_grad(set_to_none=True)
                own = rate_max_m3_s * torch.sigmoid(logits[player])
                other = rate_max_m3_s * torch.sigmoid(logits[1 - player].detach())
                payoff = _one_player_objective(
                    model, data, row, initial, own, stats, rate_scale,
                    player=player, opponent_rates=other,
                    max_rate_m3_s=rate_max_m3_s, stage_days=stage_days, rho=rho,
                    storage_value_per_kg=storage_value_per_kg,
                    pipeline_cost_per_kg=pipeline_cost_per_kg,
                    pipeline_tariff_per_kg=pipeline_tariff_per_kg,
                    storage_cap_kg=storage_cap_kg,
                    pipeline_dispatch_rule=pipeline_dispatch_rule,
                    storage_dispatch_rule=storage_dispatch_rule,
                    pressure_cap_pa=pressure_cap_pa,
                    pressure_damage_per_pa2=pressure_damage_per_pa2,
                    pressure_smoothing_pa=pressure_smoothing_pa,
                    pipeline_capacity_kg_per_stage=pipeline_capacity_kg_per_stage,
                    discount_factor=discount_factor)
                (-payoff).backward()
                optimizers[player].step()
            with torch.no_grad():
                own = rate_max_m3_s * torch.sigmoid(logits[player])
                other = rate_max_m3_s * torch.sigmoid(logits[1 - player])
                objectives[player].append(float(_one_player_objective(
                    model, data, row, initial, own, stats, rate_scale,
                    player=player, opponent_rates=other,
                    max_rate_m3_s=rate_max_m3_s, stage_days=stage_days, rho=rho,
                    storage_value_per_kg=storage_value_per_kg,
                    pipeline_cost_per_kg=pipeline_cost_per_kg,
                    pipeline_tariff_per_kg=pipeline_tariff_per_kg,
                    storage_cap_kg=storage_cap_kg,
                    pipeline_dispatch_rule=pipeline_dispatch_rule,
                    storage_dispatch_rule=storage_dispatch_rule,
                    pressure_cap_pa=pressure_cap_pa,
                    pressure_damage_per_pa2=pressure_damage_per_pa2,
                    pressure_smoothing_pa=pressure_smoothing_pa,
                    pipeline_capacity_kg_per_stage=pipeline_capacity_kg_per_stage,
                    discount_factor=discount_factor)))
    schedule = torch.stack([rate_max_m3_s * torch.sigmoid(x.detach()) for x in logits], dim=1).cpu().numpy()
    return {"rates_m3_s_by_stage_operator": schedule.tolist(),
            "objective_by_round": objectives,
            "method": "alternating open-loop differentiable best responses",
            "units": {"rate": "m3/s", "pressure": "Pa", "capacity": "kg/stage"},
            "caveat": "surrogate-planned actions; validate the complete schedule with the solver before treating as game outcomes"}


def _load_dataset(data_dir: Path, cohort: str, split: str, geology_seed: int,
                  episode_id: int | None):
    path = data_dir / split / f"{cohort}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    row = _state_at(data, geology_seed, episode_id)
    return data, row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--cohort", choices=trainer.COHORTS, required=True)
    ap.add_argument("--split", choices=("train", "validation", "test"), default="train",
                    help="use train for solver relabeling; validation/test only for held-out game-readiness checks")
    ap.add_argument("--geology-seed", type=int, required=True)
    ap.add_argument("--episode-id", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rate-max-m3-s", type=float, required=True)
    ap.add_argument("--stage-days", type=float, default=30.0)
    ap.add_argument("--pressure-cap-pa", type=float, required=True)
    ap.add_argument("--pressure-damage-per-pa2", type=float, required=True)
    ap.add_argument("--storage-value-per-kg", type=float, default=1.0)
    ap.add_argument("--pipeline-cost-per-kg", type=float, default=0.0)
    ap.add_argument("--pipeline-tariff-per-kg", type=float, default=0.0)
    ap.add_argument("--pipeline-capacity-kg-per-stage", type=float)
    ap.add_argument("--storage-cap-kg", type=float)
    ap.add_argument("--pipeline-dispatch-rule", choices=("proportional", "cea", "cel", "fixed_equal"), default="proportional")
    ap.add_argument("--storage-dispatch-rule", choices=("proportional", "cea", "cel", "fixed_equal"), default="proportional")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps-per-response", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    args = ap.parse_args()
    blob = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = trainer.make_model(blob["backbone"], blob["width"])
    model.load_state_dict(blob["state_dict"])
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in blob["stats"].items()}
    data, row = _load_dataset(args.data_dir, args.cohort, args.split,
                              args.geology_seed, args.episode_id)
    result = iterated_best_responses(
        model, data, row, stats, float(blob["rate_scale"]), rate_max_m3_s=args.rate_max_m3_s,
        stage_days=args.stage_days, pressure_cap_pa=args.pressure_cap_pa,
        pressure_damage_per_pa2=args.pressure_damage_per_pa2,
        storage_value_per_kg=args.storage_value_per_kg,
        pipeline_cost_per_kg=args.pipeline_cost_per_kg,
        pipeline_tariff_per_kg=args.pipeline_tariff_per_kg,
        storage_cap_kg=args.storage_cap_kg,
        pipeline_dispatch_rule=args.pipeline_dispatch_rule,
        storage_dispatch_rule=args.storage_dispatch_rule,
        pipeline_capacity_kg_per_stage=args.pipeline_capacity_kg_per_stage,
        rounds=args.rounds, steps_per_response=args.steps_per_response,
        learning_rate=args.learning_rate, device=args.device)
    result.update({"cohort": args.cohort, "split": args.split,
                   "geology_seed": args.geology_seed,
                   "episode_id": args.episode_id, "checkpoint": str(args.checkpoint),
                   "assumptions": {"stage_days": args.stage_days,
                                   "pressure_cap_pa": args.pressure_cap_pa,
                                   "pressure_damage_per_pa2": args.pressure_damage_per_pa2,
                                   "storage_value_per_kg": args.storage_value_per_kg,
                                   "pipeline_cost_per_kg": args.pipeline_cost_per_kg,
                                   "pipeline_tariff_per_kg": args.pipeline_tariff_per_kg,
                                   "pipeline_capacity_kg_per_stage": args.pipeline_capacity_kg_per_stage,
                                   "storage_cap_kg": args.storage_cap_kg,
                                   "pipeline_dispatch_rule": args.pipeline_dispatch_rule,
                                   "storage_dispatch_rule": args.storage_dispatch_rule}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved surrogate best-response schedule: {args.out}")


if __name__ == "__main__":
    main()
