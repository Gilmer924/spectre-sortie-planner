# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:15:31 2025

@author: scwri
"""

# simulations/rim_requirements.py

import math
from typing import Dict, Any


def compute_turn_metrics(am: int, pm: int) -> Dict[str, float]:
    """
    Turn math used in RIM / North Star:

      Turn Ratio (TR) = PM / AM
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
) -> Dict[str, Any]:
    """
    Crew-driven aircraft requirement for RIM / North Star.

    Inputs:
      pai                     = primary aircraft inventory (or TAI for now)
      crew_ratio              = crews per aircraft
      sorties_per_crew_month  = sorties each crew needs per month
      om_days                 = O&M days in the month
      turn_factor             = sorties per jet per fly day (TF)
      attrition_rate          = 0..1 (percent of sorties that fall out)

    Formula:
      Monthly net sorties from crews:
        S_net_month = crew_ratio * PAI * SPCM

      Daily net sorties:
        S_net_day = S_net_month / OM_days

      Gross lines (undo attrition):
        S_gross_day = S_net_day / (1 - attrition)

      Required aircraft:
        A_req = ceil( S_gross_day / TF )
    """

    pai        = max(0, int(pai))
    crew_ratio = max(0.0, float(crew_ratio))
    spcm       = max(0.0, float(sorties_per_crew_month))
    om_days    = max(1, int(om_days))          # no divide by zero
    tf         = max(1e-9, float(turn_factor)) # protect tiny values
    attr       = max(0.0, min(0.999999, float(attrition_rate)))

    # 1) Monthly net sorties
    net_sorties_month = crew_ratio * pai * spcm

    # 2) Daily net sorties
    daily_net_sorties = net_sorties_month / om_days

    # 3) Gross lines (undo attrition)
    daily_gross_lines = daily_net_sorties / (1.0 - attr)

    # 4) Aircraft required given TF
    aircraft_required = math.ceil(daily_gross_lines / tf)

    return {
        "pai": pai,
        "crew_ratio": crew_ratio,
        "sorties_per_crew_month": spcm,
        "om_days": om_days,
        "turn_factor": tf,
        "attrition_rate": attr,
        "net_sorties_month": net_sorties_month,
        "daily_net_sorties": daily_net_sorties,
        "daily_gross_lines": daily_gross_lines,
        "aircraft_required": aircraft_required,
    }
