# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:15:31 2025

@author: scwri
"""

# simulations/rim_requirements.py

import math
from typing import Dict, Any, List, Optional


def compute_turn_metrics(am: int, pm: int) -> Dict[str, float]:
    """
    Turn math used in RIM / North Star:

      Turn Ratio (TR)  = PM / AM
      Turn Factor (TF) = (AM + PM) / AM = 1 + TR

    TF is 'sorties per jet per fly day'.
    """
    am = max(1, int(am))  # avoid divide-by-zero
    pm = max(0, int(pm))

    turn_ratio = pm / am
    turn_factor = (am + pm) / am  # sorties per jet per day

    return {
        "am": am,
        "pm": pm,
        "turn_ratio": turn_ratio,
        "turn_factor": turn_factor,
        "sorties_per_jet": turn_factor,
    }


def compute_crew_aircraft_requirement(
    *,
    pai: int,
    crew_ratio: float,
    sorties_per_crew_month: float,
    om_days: int,
    turn_factor: float,
    attrition_rate: float,
    extra_crews: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Crew-driven aircraft requirement for RIM / North Star.

    Inputs:
      pai                     = primary aircraft inventory (or TAI for now)
      crew_ratio              = crews per aircraft (base crews)
      sorties_per_crew_month  = sorties each *base* crew needs per month
      om_days                 = O&M days in the month
      turn_factor             = sorties per jet per fly day (TF)
      attrition_rate          = 0..1 (percent of sorties that fall out)
      extra_crews             = list of dicts:
                                { "name": str, "count": float, "spcm": float }

    Logic:
      1) Compute base crews from crew_ratio × PAI (rounded UP to whole crews).
      2) Add extra crew groups (each with their own SPCM, also whole crews).
      3) Compute total monthly net sortie requirement from all crews.
      4) Convert to daily net requirement (executed sorties, after attrition).
      5) Inflate to daily gross lines (planned sorties, before attrition).
      6) Aircraft required = ceil( daily_gross_lines / TF ).

      All crews and sorties are whole numbers. No partial crews, no partial sorties.
    """

    extra_crews = extra_crews or []

    pai = max(0, int(pai))
    crew_ratio = max(0.0, float(crew_ratio))
    spcm = max(0.0, float(sorties_per_crew_month))
    om_days = max(1, int(om_days))  # no divide by zero
    tf = max(1e-9, float(turn_factor))  # protect tiny values
    attr = max(0.0, min(0.999999, float(attrition_rate)))

    # 1) Base crews (ratio × PAI → rounded up to whole crews)
    base_crews_float = crew_ratio * pai
    base_crews = math.ceil(base_crews_float) if (crew_ratio > 0 and pai > 0) else 0

    # Monthly net sorties from base crews
    base_net_sorties_month = base_crews * spcm

    # 2) Extra crew groups
    extra_crews_total = 0
    extra_net_sorties_month = 0.0

    for grp in extra_crews:
        count = max(0.0, float(grp.get("count", 0.0)))
        spcm_g = max(0.0, float(grp.get("spcm", 0.0)))
        crews_int = math.ceil(count)  # whole crews only

        if crews_int > 0 and spcm_g > 0:
            extra_crews_total += crews_int
            extra_net_sorties_month += crews_int * spcm_g

    total_crews = base_crews + extra_crews_total

    # 3) Total monthly net sortie requirement (after attrition)
    net_sorties_month = base_net_sorties_month + extra_net_sorties_month

    # 4) Daily net requirement (executed sorties, after attrition)
    #    We'll use the monthly requirement and distribute across OM days,
    #    then enforce whole sorties.
    if net_sorties_month > 0:
        daily_net_raw = net_sorties_month / om_days
        daily_net_sorties = math.ceil(daily_net_raw)
    else:
        daily_net_raw = 0.0
        daily_net_sorties = 0

    # 5) Daily gross lines (planned before attrition)
    if daily_net_sorties > 0:
        daily_gross_raw = daily_net_sorties / (1.0 - attr)
        daily_gross_lines = math.ceil(daily_gross_raw)
    else:
        daily_gross_raw = 0.0
        daily_gross_lines = 0

    # 6) Aircraft required given TF
    if daily_gross_lines > 0:
        aircraft_required = math.ceil(daily_gross_lines / tf)
    else:
        aircraft_required = 0

    return {
        "pai": pai,
        "crew_ratio": crew_ratio,
        "sorties_per_crew_month": spcm,
        "om_days": om_days,
        "turn_factor": tf,
        "attrition_rate": attr,

        # Crews
        "base_crews": float(base_crews),
        "extra_crews_total": float(extra_crews_total),
        "total_crews": float(total_crews),

        # Sorties (monthly)
        "base_net_sorties_month": float(base_net_sorties_month),
        "extra_net_sorties_month": float(extra_net_sorties_month),
        "net_sorties_month": float(net_sorties_month),

        # Sorties (daily) — integers, no partial sorties
        "daily_net_sorties": int(daily_net_sorties),     # after attrition (executed)
        "daily_gross_lines": int(daily_gross_lines),     # before attrition (planned)

        # Raw floats (in case you want them later)
        "daily_net_sorties_raw": float(daily_net_raw),
        "daily_gross_lines_raw": float(daily_gross_raw),

        # Final requirement
        "aircraft_required": int(aircraft_required),
    }
