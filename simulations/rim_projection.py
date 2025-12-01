# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:17:58 2025

@author: scwri
"""

# simulations/rim_projection.py
"""
RIM / North Star — Multi-day Projection Engine (v0.1)

This module projects EP, EP_home, and RIM over a multi-day horizon,
using a simple break/fix flow model.

It is intentionally conservative / first-order:
- Breaks are driven by a daily sorties plan and a break rate.
- Fixes are driven by a simple daily fix rate on the NMC pool.
- Depot / UPNR aircraft are treated as non-possessed (not in EP).
- RIM is computed with the same semantics you used in Databricks:
    RIM = EP_home − (Alert + Training_AM + Spares_bin)

You can call this from the Weekly Simulation or a future “RIM Projection” tab.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import math


# ---------- Core math helpers (mirrors your Databricks logic) ----------

def compute_possessed(assigned: int, non_possessed: int) -> int:
    """
    Possessed = TAI − Non-Possessed.
    """
    a = max(0, int(assigned))
    np_ = max(0, int(non_possessed))
    return max(0, a - np_)


def compute_ep(possessed: int, degraders: int) -> int:
    """
    EP = Possessed − NMC (NMCM + NMCS + NMCB).
    """
    p = max(0, int(possessed))
    d = max(0, int(degraders))
    return max(0, p - d)


def compute_ep_home(ep: int, deployed: int) -> int:
    """
    EP_home = EP − Deployed (off-station).
    """
    e = max(0, int(ep))
    dep = max(0, int(deployed))
    return max(0, e - dep)


def rim_score(ep_home: int, alert: int, training_am: int, spares_bin: int = 0) -> int:
    """
    RIM = EP_home − (Alert + Training_AM + Spares_bin)
    """
    lhs = max(0, int(ep_home))
    rhs = max(0, int(alert)) + max(0, int(training_am)) + max(0, int(spares_bin))
    return lhs - rhs


# ---------- Data structures ----------

@dataclass
class RIMProjectionInputs:
    # Structure / inventory
    tai: int                     # Total Aircraft Inventory (PAI + BAI)
    pai: int                     # Primary Aircraft Inventory
    bai: int                     # Backup Aircraft Inventory (for completeness / reporting)
    depot: int                   # Depot / Programmed (non-possessed)
    upnr: int                    # UPNR aircraft
    other_non_possessed: int = 0 # Any other non-possessed buckets

    # Degraders (NMC) at day 0
    nmcm: int = 0                # NMC Maintenance
    nmcs: int = 0                # NMC Supply
    nmcb: int = 0                # NMC Both
    nmc_flyable: int = 0         # Flyable but still NMC (does NOT count in EP)

    # Obligations
    deployed: int = 0            # Aircraft away from home station
    alert: int = 0               # Alert requirement (tails)
    spares_bin: int = 0          # Spares reserved against RIM (if any)

    # Daily operations model
    horizon_days: int = 7
    daily_training_am: Optional[List[int]] = None   # AM go tails per day
    daily_sorties: Optional[List[int]] = None       # Total sorties per day (for break math)
    break_rate_per_sortie: float = 0.16             # e.g. 16% of sorties end Code 3
    fix_rate_per_day: float = 0.40                  # fraction of NMC pool fixed per day (simple model)

    # Optional: commit cap for reference (not enforced here)
    commit_cap: float = 0.65


@dataclass
class RIMDailyState:
    day: int
    ep: int
    ep_home: int
    nmc_total: int
    nmc_flyable: int
    possessed: int
    non_possessed: int
    alert: int
    training_am: int
    spares_bin: int
    rim: int
    breaks: float
    fixes: float


# ---------- Projection engine ----------

def project_rim(inputs: RIMProjectionInputs) -> List[RIMDailyState]:
    """
    Project EP, EP_home, and RIM over `horizon_days` using a simple
    break/fix flow model.

    Assumptions:
      - Non-possessed = Depot + UPNR + other_non_possessed.
      - NMC pool = NMCM + NMCS + NMCB (NMC_flyable reported separately).
      - Daily breaks = (min(EP, sorties) * break_rate_per_sortie).
      - Daily fixes = (NMC pool * fix_rate_per_day).
      - Breaks/fixes are treated as expected values (floats) to keep this
        deterministic and fast. You can Monte-Carlo around this later.
    """
    # Precompute non-possessed
    non_poss = max(0, int(inputs.depot) + int(inputs.upnr) + int(inputs.other_non_possessed))
    tai = max(0, int(inputs.tai))

    # Validate consistency (optional, non-fatal)
    if inputs.pai + inputs.bai != tai:
        # We don't raise, but we could warn/log in a real app
        pass

    # Initial NMC pool
    nmc_pool = max(0, int(inputs.nmcm) + int(inputs.nmcs) + int(inputs.nmcb))
    nmc_fly = max(0, int(inputs.nmc_flyable))

    # Build daily plans (fallback to simple defaults if not provided)
    H = max(1, int(inputs.horizon_days))
    if inputs.daily_training_am is not None:
        if len(inputs.daily_training_am) < H:
            # pad with last value
            last = inputs.daily_training_am[-1] if inputs.daily_training_am else 0
            training_plan = inputs.daily_training_am + [last] * (H - len(inputs.daily_training_am))
        else:
            training_plan = inputs.daily_training_am[:H]
    else:
        training_plan = [0] * H  # no training if not specified

    if inputs.daily_sorties is not None:
        if len(inputs.daily_sorties) < H:
            last = inputs.daily_sorties[-1] if inputs.daily_sorties else 0
            sorties_plan = inputs.daily_sorties + [last] * (H - len(inputs.daily_sorties))
        else:
            sorties_plan = inputs.daily_sorties[:H]
    else:
        # default: assume each AM tail flies 1 sortie per day
        sorties_plan = training_plan.copy()

    brk_rate = max(0.0, float(inputs.break_rate_per_sortie))
    fix_rate = max(0.0, min(1.0, float(inputs.fix_rate_per_day)))

    results: List[RIMDailyState] = []

    # Loop over days
    current_nmc = nmc_pool
    current_nmc_fly = nmc_fly

    for d in range(1, H + 1):
        training_am = max(0, int(training_plan[d - 1]))
        daily_sorties = max(0, int(sorties_plan[d - 1]))

        # Structure for the day
        possessed = compute_possessed(tai, non_poss)
        ep = compute_ep(possessed, current_nmc)
        ep_home = compute_ep_home(ep, inputs.deployed)

        # Guard: cannot schedule more AM tails than EP_home
        training_am_effective = min(training_am, ep_home)

        # RIM score
        r = rim_score(
            ep_home=ep_home,
            alert=inputs.alert,
            training_am=training_am_effective,
            spares_bin=inputs.spares_bin
        )

        # Break/fix flows (expected values)
        # Only EP_home tails can fly
        max_possible_flyers = max(0, ep_home)
        effective_sorties = min(daily_sorties, max_possible_flyers)  # simple model: 1 sortie per flyer
        expected_breaks = effective_sorties * brk_rate
        expected_fixes = current_nmc * fix_rate

        # Update NMC pool for next day (keep non-negative and <= possessed)
        next_nmc = current_nmc + expected_breaks - expected_fixes
        next_nmc = max(0.0, min(float(possessed), next_nmc))

        # For now, keep NMC_flyable constant (can be modeled later)
        next_nmc_fly = current_nmc_fly

        results.append(
            RIMDailyState(
                day=d,
                ep=int(ep),
                ep_home=int(ep_home),
                nmc_total=int(round(current_nmc)),
                nmc_flyable=int(next_nmc_fly),
                possessed=int(possessed),
                non_possessed=int(non_poss),
                alert=int(inputs.alert),
                training_am=int(training_am_effective),
                spares_bin=int(inputs.spares_bin),
                rim=int(r),
                breaks=float(expected_breaks),
                fixes=float(expected_fixes),
            )
        )

        # advance to next day
        current_nmc = next_nmc
        current_nmc_fly = next_nmc_fly

    return results


# ---------- Convenience: export as dicts for Streamlit / plotting ----------

def project_rim_as_dicts(inputs: RIMProjectionInputs) -> List[Dict]:
    """
    Helper to get the projection as a list of plain dicts, which is
    convenient for Pandas / Plotly / Streamlit.
    """
    return [state.__dict__ for state in project_rim(inputs)]
