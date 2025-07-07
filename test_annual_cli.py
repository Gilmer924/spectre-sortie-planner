# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 07:26:10 2025

@author: scwri
"""

import pandas as pd
from simulations.historical_annual import HistoricalAnnualSimulation
from simulations.utils import calculate_monthly_rates

# 1) Load your data
df = pd.read_excel(r"C:\Users\scwri\OneDrive\Desktop\sortie projection tool upload 1.xlsx", sheet_name="Historical Data Input")
rates_df = calculate_monthly_rates(df)

# 2) Dummy user inputs (fill in for each month)
TAI = 12
turn_patterns = {m: "6x4" for m in range(1,13)}
commit_rates  = {m: 0.8   for m in range(1,13)}
om_days       = {m: 20    for m in range(1,13)}
degraders     = {m: 2     for m in range(1,13)}

params = {
    "TAI": TAI,
    "rates_df": rates_df,
    "om_days": om_days,
    "planned_degraders": degraders,
    "turn_patterns": turn_patterns,
    "commit_rates": commit_rates,
}

sim = HistoricalAnnualSimulation()
sim.params = params
results = sim.simulate(trials=1)[0]

for r in results:
    print(r)
