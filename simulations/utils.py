# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 06:31:44 2025

@author: Gilmer Jenkins
"""
import pandas as pd

def calculate_monthly_rates(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Takes normalized df_hist and computes monthly rates used by the Annual Historical Simulation.

    Expected minimum columns (case/spacing doesn't matter if you normalized earlier):
      - month
      - tai_hours
      - possessed_hours
      - mc_hours
      - hours_flown
      - sorties_flown
      - sorties_scheduled
      - ground_aborts
      - not_spared_ground_aborts
      - breaks
      - fixes_8hr
      - fixes_12hr
      - fixes_24hr

    Returns:
      DataFrame with month_num plus calculated rates and average aircraft counts.
    """
    import calendar

    df = df_hist.copy()

    # --- Normalize month to month_num ---
    name2num = {name: idx for idx, name in enumerate(calendar.month_name) if name}
    df["month_num"] = (
        df["month"]
          .astype(str)
          .str.strip()
          .str.title()
          .map(name2num)
    )

    # Defensive: drop rows where month is invalid/unmapped
    df = df[~df["month_num"].isna()].copy()
    df["month_num"] = df["month_num"].astype(int)

    # --- Days in month (use canonical average for Feb to support multi-year rollups) ---
    canonical_days = {
        1: 31, 2: 28.25, 3: 31, 4: 30,
        5: 31, 6: 30, 7: 31, 8: 31,
        9: 30, 10: 31, 11: 30, 12: 31
    }
    df["days_in_month"] = df["month_num"].map(canonical_days).astype(float)

    # --- Helper for safe division (avoids divide-by-zero -> inf/NaN explosions) ---
    def safe_div(numer, denom):
        denom = denom.replace(0, pd.NA) if isinstance(denom, pd.Series) else denom
        return (numer / denom)

    # --- Core sortie rates ---
    df["execution_rate"]  = safe_div(df["sorties_flown"], df["sorties_scheduled"])
    df["attrition_rate"]  = 1.0 - df["execution_rate"]

    df["gab_rate"]        = safe_div(df["ground_aborts"], df["sorties_scheduled"])
    df["spared_gab_rate"] = safe_div(df["not_spared_ground_aborts"], df["sorties_scheduled"])

    df["break_rate"]      = safe_div(df["breaks"], df["sorties_scheduled"])

    # --- Fix rates (based on flown sorties) ---
    df["fix_rate_8hr"]    = safe_div(df["fixes_8hr"], df["sorties_flown"])
    df["fix_rate_12hr"]   = safe_div(df["fixes_12hr"], df["sorties_flown"])
    df["fix_rate_24hr"]   = safe_div(df["fixes_24hr"], df["sorties_flown"])

    # --- Health rates (MC and AA) ---
    # MC rate: within possessed
    df["mc_rate"]         = safe_div(df["mc_hours"], df["possessed_hours"])

    # AA rate: within assigned (TAI_hours)
    # This is the key new piece you asked for.
    df["aa_rate"]         = safe_div(df["mc_hours"], df["tai_hours"])

    # --- Average sortie duration ---
    df["asd"]             = safe_div(df["hours_flown"], df["sorties_flown"])

    # --- Average aircraft counts (hours -> tails) ---
    denom_hours = df["days_in_month"] * 24.0

    df["avg_assigned_ac"] = safe_div(df["tai_hours"], denom_hours)
    df["avg_poss_ac"]     = safe_div(df["possessed_hours"], denom_hours)
    df["avg_fly_ac"]      = safe_div(df["mc_hours"], denom_hours)

    # --- Clean up: clamp common rates to valid bounds where appropriate ---
    # (These can still be NA if inputs are missing/0; that's OK.)
    for col in ["execution_rate", "mc_rate", "aa_rate"]:
        df[col] = df[col].clip(lower=0, upper=1)

    # --- Optional rounding for display/editing convenience ---
    round_cols = [
        "execution_rate", "attrition_rate",
        "gab_rate", "spared_gab_rate", "break_rate",
        "fix_rate_8hr", "fix_rate_12hr", "fix_rate_24hr",
        "mc_rate", "aa_rate", "asd",
        "avg_assigned_ac", "avg_poss_ac", "avg_fly_ac",
    ]
    for col in round_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    return df
