# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 06:31:44 2025

@author: Gilmer Jenkins
"""
import pandas as pd
import numpy as np
import calendar

def calculate_monthly_rates(df_hist):

    """
    Compute monthly historical rates used by Annual Historical Simulation.
    Restored with Numeric Coercion, Abbreviation-Safe Mapping, and Gap Patch.
    """

    df = df_hist.copy()

    # --- 1) RESTORED: Abbreviation-Safe Month Mapping ---
    # Catch both "Dec" and "December" to prevent quiet row deletion
    full_names = {name: idx for idx, name in enumerate(calendar.month_name) if name}
    abbr_names = {name: idx for idx, name in enumerate(calendar.month_abbr) if name}
    combined_map = {**full_names, **abbr_names}

    df["month_num"] = (
        df["month"]
          .astype(str)
          .str.strip()
          .str.title()
          .map(combined_map)
    )

    # Defensive: drop rows where month is invalid/unmapped
    df = df[~df["month_num"].isna()].copy()
    df["month_num"] = df["month_num"].astype(int)

    # --- 2) RESTORED: Numeric Coercion ---
    # Critical: Converts "7,440" (string) to 7440 (float) so math works
    numeric_cols = [
        "tai_hours", "possessed_hours", "mc_hours", "hours_flown",
        "sorties_flown", "sorties_scheduled",
        "ground_aborts", "not_spared_ground_aborts",
        "breaks", "fixes_8hr", "fixes_12hr", "fixes_24hr",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Days in month mapping ---
    canonical_days = {
        1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    df["days_in_month"] = df["month_num"].map(canonical_days).astype(float)

    # --- Improved Safe Division Helper ---
    def safe_div(numer, denom):
        # Returns NaN instead of Inf if denom is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(denom == 0, np.nan, numer / denom)
        return pd.Series(out, index=df.index, dtype="float64")

    # --- Core Calculations ---
    df["execution_rate"]  = safe_div(df["sorties_flown"], df["sorties_scheduled"])
    df["attrition_rate"]  = 1.0 - df["execution_rate"].fillna(0) # Logic fix: if exec is NaN, attrition is effectively 0

    df["gab_rate"]        = safe_div(df["ground_aborts"], df["sorties_scheduled"])
    df["spared_gab_rate"] = safe_div(df["not_spared_ground_aborts"], df["sorties_scheduled"])
    df["break_rate"]      = safe_div(df["breaks"], df["sorties_scheduled"])

    # How many of the BROKEN aircraft were fixed in X time?
    df["fix_rate_8hr"]  = safe_div(df["fixes_8hr"], df["breaks"])
    df["fix_rate_12hr"] = safe_div(df["fixes_12hr"], df["breaks"])
    df["fix_rate_24hr"] = safe_div(df["fixes_24hr"], df["breaks"])
    
    # Anything left over is a "Long Fix" or "Non-Flyable"
    df["long_fix_rate"] = 1.0 - (df["fix_rate_8hr"] + df["fix_rate_12hr"] + df["fix_rate_24hr"])

    # Health rates (MC and AA)
    df["mc_rate"]         = safe_div(df["mc_hours"], df["possessed_hours"])
    df["aa_rate"]         = safe_div(df["mc_hours"], df["tai_hours"])
    df["asd"]             = safe_div(df["hours_flown"], df["sorties_flown"])

    # --- 3) NEW PATCH: Gap Prevention ---
    # If a month is missing data, use the unit's annual average 
    # so the charts don't show "Gaps" or 0%
    for col in ["mc_rate", "asd", "execution_rate"]:
        avg_val = df[col].mean()
        df[col] = df[col].fillna(avg_val if pd.notna(avg_val) else 0)

    # --- Tails Calculation ---
    denom_hours = df["days_in_month"] * 24.0
    df["avg_assigned_ac"] = safe_div(df["tai_hours"], denom_hours)
    df["avg_poss_ac"]     = safe_div(df["possessed_hours"], denom_hours)
    df["avg_fly_ac"]      = safe_div(df["mc_hours"], denom_hours)

    # --- Final Polish ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for col in ["execution_rate", "mc_rate", "aa_rate"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=1)

    round_cols = [
        "execution_rate", "attrition_rate", "gab_rate", "spared_gab_rate", 
        "break_rate", "fix_rate_8hr", "fix_rate_12hr", "fix_rate_24hr",
        "mc_rate", "aa_rate", "asd", "avg_assigned_ac", "avg_poss_ac", "avg_fly_ac"
    ]
    for col in round_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    return df
