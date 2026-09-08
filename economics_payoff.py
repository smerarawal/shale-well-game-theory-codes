"""
economics_payoff.py -- Module 4.1 from implementation_spec_v3.md.

HONESTY NOTE: the spec assumes a trajectory FNO (full pressure history,
not just final frame) already exists so this could compute a proper
time-integrated discounted EUR. That FNO was never built in this repo --
only the terminal-pressure FNO (optimize_wells.py's predict_pressure)
exists. So this implements the best available version: converts the
EXISTING terminal-pressure proxy payoff into real $/ton units using the
benchmark paper's economic parameters, discounted as a ONE-PERIOD payoff
(not a true multi-period NPV, since there's no per-period trajectory to
discount over yet). This is explicitly a partial implementation of 4.1 --
full sequential economics needs the trajectory FNO (spec Module 1.4/3.1)
built first. Flagged here rather than silently overclaiming.

Once a trajectory FNO exists, replace `terminal_payoff_to_dollars` below
with a proper per-period sum using the real pressure history and this
same $/ton conversion -- the economics constants and conversion logic
don't change, only the time-integration does.

Run:
    python economics_payoff.py

Expects dataset_2000.npz and fno_surrogate.pt in the same directory
(reuses optimize_wells.py's trained model).
"""
import numpy as np
import torch

from optimize_wells import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32

# Economic parameters -- matching the benchmark paper's Table 2 for direct
# comparability, per the spec
R_CREDIT = 85.0        # $/ton tax credit
R_OP = 45.0            # $/ton capture/transport/storage cost
NET_MARGIN = R_CREDIT - R_OP  # $20.0/ton net margin
DISCOUNT_RATE = 0.05   # annual discount rate (explicit: NOT the paper's 0.95
                        # per-period factor -- that already bakes in a period
                        # length; 0.05 annual is used here since this is a
                        # single-snapshot payoff, not per-period, so the two
                        # aren't directly interchangeable -- flagged, don't
                        # silently treat them as the same number)

# Conversion factor from the simplified pressure-based payoff proxy
# (arbitrary simulation units) to tons -- THIS IS A PLACEHOLDER SCALING
# CONSTANT, not derived from real reservoir volumetrics. Document prominently
# if used in any report: it makes units consistent for the audit in
# audit_core_stability.py, it does NOT make the payoff physically calibrated.
PROXY_TO_TONS_SCALE = 1000.0


def terminal_payoff_to_dollars(proxy_payoff, discount_periods=0):
    """
    Convert the existing simplified proxy payoff (sum of |pressure| at
    well locations) into a dollar figure using the benchmark's $/ton
    parameters. See module docstring: this is a ONE-SHOT conversion, not
    a true discounted multi-period NPV, since no trajectory data exists
    yet to discount over time.
    """
    tons = proxy_payoff * PROXY_TO_TONS_SCALE
    dollars = tons * NET_MARGIN
    discount_factor = 1.0 / ((1 + DISCOUNT_RATE) ** discount_periods)
    return dollars * discount_factor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats()
    model = load_fno(device=device)

    np.random.seed(11)
    perm = random_permeability_field(NX, NY)

    agent_wells = {
        "A": [(8, 8), (10, 10)],
        "B": [(9, 9)],
        "C": [(20, 20), (22, 22), (20, 24)],
    }

    print(f"Economic parameters: R_credit=${R_CREDIT}/ton  R_op=${R_OP}/ton  "
          f"net_margin=${NET_MARGIN}/ton  discount_rate={DISCOUNT_RATE}")
    print(f"NOTE: proxy-to-tons scale ({PROXY_TO_TONS_SCALE}) is a placeholder, "
          "not physically calibrated -- see module docstring.\n")

    for agent, locs in agent_wells.items():
        rates = [WELL_RATE] * len(locs)
        pressure = predict_pressure(model, perm, locs, rates, norm_stats, device)
        proxy = total_payoff(pressure, locs)
        dollars = terminal_payoff_to_dollars(proxy)
        print(f"agent {agent}: proxy_payoff={proxy:.3f}  ->  ${dollars:,.0f}")

    print("\nThis gives every existing Shapley/core/nucleolus/Blotto script a "
          "drop-in dollar-unit payoff (call terminal_payoff_to_dollars() on "
          "whatever total_payoff() currently returns) -- but it is a UNIT "
          "CONVERSION of the same simplified proxy, not the richer "
          "sequential/discounted economics the spec's Module 4.1 actually "
          "describes. That needs the trajectory FNO built first.")


if __name__ == "__main__":
    main()
