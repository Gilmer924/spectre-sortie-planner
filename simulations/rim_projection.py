# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:17:58 2025

@author: scwri
"""

# simulations/rim_projections.py
"""
RIM / North Star — Multi-day Projection Engine (v0.1)

This module projects EP, EP_home, and RIM over a multi-day horizon, using a simple
break/fix flow model.

It mirrors the Databricks RIM math (EP, EP_home, RIM) and adds deterministic
daily break/fix behavior so you can evaluate how training patterns and fix rates
shape readiness over time.

You can wire this into the Weekly Simulation or create a standalone RIM Projection tab.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import math


# ============================================================
# Core math helpers (mirrors your Databricks logic)
# ============================================================

def compute_possessed(assigned: int, non_possessed: int) -> int:
    """Possessed = TAI − Non-Possessed."""
    a = max(0, int(assigned))
    np_ = max(0, int(non_possessed))
    return max(0, a - np_)


def compute_ep(possessed: int, degraders: int) -> int:
    """EP = Possessed − NMC (NMCM + NMCS + NMCB)."""
    p = max(0, int(possessed))
    d = max(0, int(degraders))
    return max(0, p - d)


def compute_ep_home(ep: int, deployed: int) -> int:
    """EP_home = EP − Deployed (off-station)."""
    e = max(0, int(ep))
    dep = max(0, int(deployed))
    return max(0, e - dep)


def rim_score(ep_home: int, alert: int, training_am: int, spares_bin: int = 0) -> int:
    """RIM = EP_home − (Alert + Training_AM + Spares_bin)."""
    lhs = max(0, int(ep_home))
    rhs = max(0, int(alert)) + max(0, int(training_am)) + max(0, int(spares_bin))
    return lhs - rhs


# ============================================================
# Data structures
# ============================================================

@dataclass
class RIMProjectionInputs:
    # ----- Structure / inventory -----
    tai: int                     # Total Aircraft Inventory (PAI + BAI)
    pai: int                     # Primary Aircraft Inventory
    bai: int                     # Backup Aircraft Inventory
    depot: int                   # Depot aircraft (non-possessed)
    upnr: int                    # UPNR aircraft (non-possessed awaiting disposition)
    other_non_possessed: int = 0 # Any other non-possessed counts

    # ----- Degraders (NMC) state at day 0 -----
    nmcm: int = 0                # NMC Maintenance
    nmcs: int = 0                # NMC Supply
    nmcb: int = 0                # NMC Both
    nmc_flyable: int = 0         # Flyable but still NMC (adds headroom, not EP)

    # ----- Obligations -----
    deployed: int = 0            # Aircraft away from home station
    alert: int = 0               # Alert requirement (tails)
    spares_bin: int = 0          # Reserved spares (counted against RIM)

    # ----- Daily operations -----
    horizon_days: int = 7
    daily_training_am: Optional[List[int]] = None
    daily_sorties: Optional[List[int]] = None
    break_rate_per_sortie: float = 0.16  # Default: 16% Code-3 rate
    fix_rate_per_day: float = 0.40       # Fraction of NMC pool fixed per day

    # Optional: commit cap reference (not enforced here)
    commit_cap: float = 0.65


@dataclass
class RIMDailyState:
    """Daily record for EP / RIM evolution."""
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


# ============================================================
# Projection engine
# ============================================================

def project_rim(inputs: RIMProjectionInputs) -> List[RIMDailyState]:
    """
    Project EP, EP_home, and RIM over horizon_days, modeling breaks and fixes.

    Assumptions:
    - NMC pool = NMCM + NMCS + NMCB (NMC_flyable kept separate)
    - Breaks = min(EP_home, sorties) × break_rate
    - Fixes = NMC pool × fix_rate
    - All flows treated as expected-value (deterministic)
    """

    # ----- Compute non-possessed -----
    non_poss = max(0, inputs.depot + inputs.upnr + inputs.other_non_possessed)
    tai = max(0, inputs.tai)

    # ----- Initial NMC pools -----
    nmc_pool = max(0, inputs.nmcm + inputs.nmcs + inputs.nmcb)
    nmc_fly = max(0, inputs.nmc_flyable)

    # ----- Harmonize daily vectors -----
    H = max(1, inputs.horizon_days)

    # Training AM lines
    if inputs.daily_training_am is None:
        training_plan = [0] * H
    else:
        if len(inputs.daily_training_am) < H:
            last = inputs.daily_training_am[-1]
            training_plan = inputs.daily_training_am + [last] * (H - len(inputs.daily_training_am))
        else:
            training_plan = inputs.daily_training_am[:H]

    # Daily total sorties
    if inputs.daily_sorties is None:
        sorties_plan = training_plan.copy()  # assume 1 sortie per AM tail
    else:
        if len(inputs.daily_sorties) < H:
            last = inputs.daily_sorties[-1]
            sorties_plan = inputs.daily_sorties + [last] * (H - len(inputs.daily_sorties))
        else:
            sorties_plan = inputs.daily_sorties[:H]

    brk_rate = max(0.0, float(inputs.break_rate_per_sortie))
    fix_rate = max(0.0, min(1.0, float(inputs.fix_rate_per_day)))

    results: List[RIMDailyState] = []
    current_nmc = float(nmc_pool)
    current_nmc_fly = float(nmc_fly)

    # ============================================================
    # Daily loop
    # ============================================================

    for d in range(1, H + 1):
        training_am = max(0, int(training_plan[d - 1]))
        daily_sorties = max(0, int(sorties_plan[d - 1]))

        # ----- Structure for the day -----
        possessed = compute_possessed(tai, non_poss)
        ep = compute_ep(possessed, int(round(current_nmc)))
        ep_home = compute_ep_home(ep, inputs.deployed)

        # Guard: training AM cannot exceed EP_home
        training_am_eff = min(training_am, ep_home)

        # ----- RIM score -----
        r = rim_score(
            ep_home=ep_home,
            alert=inputs.alert,
            training_am=training_am_eff,
            spares_bin=inputs.spares_bin
        )

        # ----- Breaks and fixes -----
        max_possible_flyers = max(0, ep_home)
        effective_sorties = min(daily_sorties, max_possible_flyers)
        expected_breaks = effective_sorties * brk_rate
        expected_fixes = current_nmc * fix_rate

        # ----- Update NMC pool -----
        next_nmc = current_nmc + expected_breaks - expected_fixes
        next_nmc = max(0.0, min(float(possessed), next_nmc))

        # Keep NMC_flyable constant for now
        next_nmc_fly = current_nmc_fly

        # ----- Record day -----
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
                training_am=int(training_am_eff),
                spares_bin=int(inputs.spares_bin),
                rim=int(r),
                breaks=float(expected_breaks),
                fixes=float(expected_fixes),
            )
        )

        # ----- Advance -----
        current_nmc = next_nmc
        current_nmc_fly = next_nmc_fly

    return results


# ============================================================
# Convenience for Streamlit / Pandas
# ============================================================

def project_rim_as_dicts(inputs: RIMProjectionInputs) -> List[Dict]:
    """
    Return the projection as a list of plain dicts—easy for DataFrame
    conversion or Plotly visualizations.
    """
    return [state.__dict__ for state in project_rim(inputs)]

