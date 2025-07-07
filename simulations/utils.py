# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 06:31:44 2025

@author: scwri
"""
import pandas as pd

def calculate_monthly_rates(df_hist):
    """
    Takes your normalized df_hist and computes all key rates needed for the annual sim.
    Expects columns: month, mc_hours, possessed_hours, hours_flown, sorties_flown, sorties_scheduled,
    ground_aborts, not_spared_ground_aborts, breaks, fixes_8hr, fixes_12hr, fixes_24hr, etc.
    Returns a new DataFrame with month_num and calculated rates.
    """
    import calendar
    # Normalize headers (should already be, but just in case)
    df = df_hist.copy()
    name2num = {name: idx for idx, name in enumerate(calendar.month_name) if name}
    df["month_num"] = (
        df["month"]
          .astype(str)
          .str.strip()
          .str.title()
          .map(name2num)
    )
    canonical_days = {
        1: 31, 2: 28.25, 3: 31, 4: 30,
        5: 31, 6: 30, 7: 31, 8: 31,
        9: 30, 10: 31, 11: 30, 12: 31
    }
    df["days_in_month"] = df["month_num"].map(canonical_days)
    # Core rates
    df["execution_rate"]      = df["sorties_flown"] / df["sorties_scheduled"]
    df["attrition_rate"]      = 1.0 - df["execution_rate"]
    df["gab_rate"]            = df["ground_aborts"] / df["sorties_scheduled"]
    df["spared_gab_rate"]     = df["not_spared_ground_aborts"] / df["sorties_scheduled"]
    df["break_rate"]          = df["breaks"] / df["sorties_scheduled"]
    df["fix_rate_8hr"]        = df["fixes_8hr"] / df["sorties_flown"]
    df["fix_rate_12hr"]       = df["fixes_12hr"] / df["sorties_flown"]
    df["fix_rate_24hr"]       = df["fixes_24hr"] / df["sorties_flown"]
    df["mc_rate"]             = df["mc_hours"] / df["possessed_hours"]
    df["asd"]                 = df["hours_flown"] / df["sorties_flown"]
    # Avgs
    df["avg_poss_ac"] = df["possessed_hours"] / (df["days_in_month"] * 24.0)
    df["avg_fly_ac"]  = df["mc_hours"] / (df["days_in_month"] * 24.0)
    # Round if desired
    for col in [
        "execution_rate","attrition_rate","gab_rate","spared_gab_rate","break_rate",
        "fix_rate_8hr","fix_rate_12hr","fix_rate_24hr","mc_rate","asd",
        "avg_poss_ac","avg_fly_ac"
    ]:
        df[col] = df[col].round(3)
    return df
