# -*- coding: utf-8 -*-

"""
Created on Tue Jun 24 2025

@author: Gilmer Jenkins
"""
# simulations/personnel_simulation.py

import calendar
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from simulations.simulation_base import SimulationBase

TEMPLATE_COLUMNS = [
    "shop",
    "auth_3", "auth_5", "auth_7",
    "asgn_3", "asgn_5", "asgn_7",
]

def workcenters_to_template_df(workcenters: Dict[str, Dict[str, float]],
                               auth: Optional[Dict[str, Dict[str, float]]] = None):
    """
    Build a template DataFrame for user download.

    workcenters: {shop: {"3":count,"5":count,"7":count}}  (assigned)
    auth:        {shop: {"3":count,"5":count,"7":count}}  (authorized) optional
    """
    if pd is None:
        raise RuntimeError("pandas is required to build the template dataframe")

    rows = []
    for shop, levels in (workcenters or {}).items():
        row = {
            "shop": shop,
            "asgn_3": float(levels.get("3", 0.0)),
            "asgn_5": float(levels.get("5", 0.0)),
            "asgn_7": float(levels.get("7", 0.0)),
            "auth_3": float((auth or {}).get(shop, {}).get("3", 0.0)),
            "auth_5": float((auth or {}).get(shop, {}).get("5", 0.0)),
            "auth_7": float((auth or {}).get(shop, {}).get("7", 0.0)),
        }
        rows.append(row)

    # If empty, return a few blank example rows to guide users
    if not rows:
        rows = [
            {"shop": "Crew Chiefs", "auth_3": 0, "auth_5": 0, "auth_7": 0, "asgn_3": 0, "asgn_5": 0, "asgn_7": 0},
            {"shop": "Avionics",    "auth_3": 0, "auth_5": 0, "auth_7": 0, "asgn_3": 0, "asgn_5": 0, "asgn_7": 0},
            {"shop": "Engines",     "auth_3": 0, "auth_5": 0, "auth_7": 0, "asgn_3": 0, "asgn_5": 0, "asgn_7": 0},
        ]

    df = pd.DataFrame(rows)
    return df[TEMPLATE_COLUMNS]

def template_df_to_workcenters(df) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Parse a user-uploaded template DataFrame into:
      - assigned dict: {shop: {"3":count,"5":count,"7":count}}
      - authorized dict: same structure
    """
    if pd is None:
        raise RuntimeError("pandas is required to parse the template dataframe")

    if df is None or df.empty:
        return {}, {}

    # normalize columns
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = set(TEMPLATE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Template missing columns: {', '.join(sorted(missing))}")

    df["shop"] = df["shop"].astype(str).str.strip()
    df = df[df["shop"] != ""]

    for c in ["auth_3","auth_5","auth_7","asgn_3","asgn_5","asgn_7"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    assigned = {}
    authorized = {}
    for _, r in df.iterrows():
        shop = r["shop"]
        assigned[shop] = {"3": float(r["asgn_3"]), "5": float(r["asgn_5"]), "7": float(r["asgn_7"])}
        authorized[shop] = {"3": float(r["auth_3"]), "5": float(r["auth_5"]), "7": float(r["auth_7"])}

    # Optional: drop shops with zero assigned across all levels
    assigned = {s: lv for s, lv in assigned.items() if sum(lv.values()) > 0}

    return assigned, authorized


class PersonnelSimulation(SimulationBase):
    """
    Monte Carlo or deterministic simulation of sortie-producing labor capacity.
    
    Tracks:
        - present people after leave/TDY/deploy (accounts for overlap)
        - available person-hours (UTE × O&M days)
        - sorties_supported (hours ÷ labor_hours_per_sortie)
        - ppl_per_ac (people per aircraft)
        - shifts_supported (8-hr shifts per AC)
        - shortfall flag vs. monthly goal
        - bottleneck shop/skill if a shortfall occurs
    """

    def validate_params(self) -> None:
        """Validate simulation input parameters, with informative error messages."""
        p = self.params
        tai = p.get("TAI")
        if not isinstance(tai, (int, float)) or tai <= 0:
            raise ValueError(f"‘TAI’ must be a positive number, got {tai!r}")

        months = p.get("months")
        if not isinstance(months, list) or not months:
            raise ValueError("‘months’ must be a non-empty list of month numbers")

        omd = p.get("om_days")
        if not isinstance(omd, dict):
            raise ValueError("‘om_days’ must be a dict month→O&M-days")

        goals = p.get("month_goals")
        if not isinstance(goals, dict):
            raise ValueError("‘month_goals’ must be a dict month→goal")

        for key in ("leave_rate", "tdy_rate", "deploy_rate"):
            v = p.get(key, 0.0)
            if not (isinstance(v, (int, float)) and 0 <= v <= 1):
                raise ValueError(f"{key!r} must be between 0 and 1 (got {v!r})")

        ute = p.get("ute_rates")
        if not isinstance(ute, dict) or not ute:
            raise ValueError("‘ute_rates’ must be a dict level→hours_per_day")

        lps = p.get("labor_hours_per_sortie")
        if not (isinstance(lps, (int, float)) and lps > 0):
            raise ValueError("‘labor_hours_per_sortie’ must be positive (got {lps!r})")

        wcs = p.get("workcenters")
        if not isinstance(wcs, dict) or not wcs:
            raise ValueError("‘workcenters’ must be a non-empty dict shop→{level:count}")

    def _calculate_absence(self, leave: float, tdy: float, deploy: float) -> float:
        """
        Calculate the combined probability of absence (not just sum)
        - returns 1 - probability of present
        """
        present_prob = (1 - leave) * (1 - tdy) * (1 - deploy)
        return 1 - present_prob

    def simulate(self, trials: int = 1) -> List[List[Dict[str, Any]]]:
            """
            Refactored Simulation: Tracks Wrench-Turners vs. Present, 
            Out-of-Hide noise, and Authorized/Assigned health.
            """
            self.validate_params()
            p = self.params
            months = p["months"]
            omd = p["om_days"]
            goals = p["month_goals"]
            ute = p["ute_rates"]
            lps = float(p["labor_hours_per_sortie"])
            wcs = p["workcenters"]
            auth_wcs = p.get("auth_workcenters", {})  # New expected param
            tai = float(p["TAI"])
            
            # New: Monthly drain map (Leave, OOM, Deploy)
            # Fallback to static sidebar rates if monthly map isn't provided
            monthly_drain = p.get("monthly_drain", {})
    
            stochastic = (trials > 1)
            all_trials = []
            this_year = datetime.now().year
    
            for _ in range(trials):
                trial_months = []
                for m in months:
                    days_in_month = omd.get(m, calendar.monthrange(this_year, m)[1])
                    
                    # Fetch month-specific drains or use global defaults
                    m_drain = monthly_drain.get(m, {
                        "leave": p.get("leave_rate", 0.10),
                        "tdy": p.get("tdy_rate", 0.05),
                        "deploy": p.get("deploy_rate", 0.02),
                        "oom": 0.10  # Default 10% Out-of-Hide/Noise
                    })
    
                    # Calculate probability of being "Present" (not on Leave/TDY/Deploy)
                    presence_rate = (1 - m_drain["leave"]) * (1 - m_drain.get("tdy", 0)) * (1 - m_drain["deploy"])
                    oom_rate = m_drain.get("oom", 0.0) # Out-of-Hide noise
    
                    total_assigned = 0.0
                    total_auth = 0.0
                    total_present = 0.0
                    total_wrench_turners = 0.0
                    total_hours = 0.0
    
                    shop_details = {}
    
                    for shop, levels in wcs.items():
                        shop_asgn = 0.0
                        shop_auth = sum(auth_wcs.get(shop, {}).values()) if auth_wcs else 0.0
                        shop_present = 0.0
                        shop_wt = 0.0 # Shop Wrench Turners
                        shop_hrs = 0.0
    
                        for lvl, asgn in levels.items():
                            shop_asgn += asgn
                            
                            # Calculate Present vs Wrench Turner
                            if stochastic:
                                # Binomial selection for presence
                                n_present = np.random.binomial(int(asgn), min(1, presence_rate))
                            else:
                                n_present = asgn * presence_rate
                            
                            # Subtract Noise (Out-of-Hide tasks like CTK/CSS)
                            # Noise is calculated against assigned strength
                            n_wt = max(0, n_present - (asgn * oom_rate))
                            
                            shop_present += n_present
                            shop_wt += n_wt
                            shop_hrs += n_wt * ute.get(lvl, 0.0) * days_in_month
    
                        total_assigned += shop_asgn
                        total_auth += shop_auth
                        total_present += shop_present
                        total_wrench_turners += shop_wt
                        total_hours += shop_hrs
    
                        shop_details[shop] = {
                            "asgn": shop_asgn,
                            "auth": shop_auth,
                            "present": shop_present,
                            "wrench_turners": shop_wt,
                            "hrs_avail": shop_hrs,
                            "hrs_req": goals.get(m, 0) * (shop_hrs / total_hours) if total_hours > 0 else 0
                        }
    
                    sorties_supported = total_hours / lps if lps > 0 else 0
                    shortfall = sorties_supported < goals.get(m, 0)
    
                    # Find limiting shop
                    limiting_shop = ""
                    limiting_shop_sorties = None
                    if shortfall:
                        # Capacity logic: which shop's wrench-turner hours fail first?
                        shop_cap = {s: d["hrs_avail"]/lps if lps > 0 else 0 for s, d in shop_details.items()}
                        limiting_shop, limiting_shop_sorties = min(shop_cap.items(), key=lambda x: x[1])
    
                    trial_months.append({
                        "month": m,
                        "assigned_total": total_assigned,
                        "auth_total": total_auth,
                        "manning_health": total_assigned / total_auth if total_auth > 0 else 0,
                        "present_people": total_present,
                        "wrench_turners": total_wrench_turners,
                        "available_hours": total_hours,
                        "sorties_supported": sorties_supported,
                        "shortfall": shortfall,
                        "limiting_shop": limiting_shop if shortfall else "",
                        "limiting_shop_sorties": limiting_shop_sorties if shortfall else None,
                        "ppl_per_ac": total_wrench_turners / tai if tai > 0 else 0,
                        "leave_rate": float(m_drain.get("leave", 0.0)),
                        "tdy_rate": float(m_drain.get("tdy", 0.0)),
                        "deploy_rate": float(m_drain.get("deploy", 0.0)),
                        "oom_rate": float(m_drain.get("oom", 0.0)),
                        "shop_details": shop_details  # Critical for the Heatmap
                    })
                all_trials.append(trial_months)
            return all_trials

    def run(self) -> List[Dict[str, Any]]:
        """
        Deterministic single-trial shortcut
        Returns: list of monthly results (for one trial)
        """
        return self.simulate(trials=1)[0]

