# spectre-sortie-planner
SPECTRE: Sortie Planning &amp; Simulation A Streamlit app for Monte Carlo driven sortie, maintenance, and personnel capacity modeling.

## July 2025 Major Update

### New Features
- **Editable Input Matrix:** Easily configure O&M days, degraders, turn patterns, commit rates, and ASD for each month in a single grid.
- **Historical Rates Editing:** Review or override calculated monthly rates before simulation.
- **Monte Carlo Simulation:** Adds random variation for robust forecasts; customize uncertainty level.
- **Goal Probability:** See the estimated probability of meeting your annual Flying Hour Program goal.
- **Detailed Results Table:** Includes monthly/summed sorties, hours, and totals for download.
- **Alerts & Warnings:** Commitment risks now display in-app (no longer just in terminal).
- **Clean UI:** Planning, alerts, and rates in collapsible expanders for easier navigation.

### How to Use
1. **Upload**: Drag/drop your 4-year Excel roll-up (sheet "Historical Data Input").
2. **Plan**: Edit the FY Planning matrix and (if needed) adjust historical rates.
3. **Simulate**: Set MC targets, ASD, MC-delta, and uncertainty, then run analysis.
4. **Review**: Check warnings, probability chart, monthly results, and download CSV.

### Developer Notes
- Core simulation code in `simulations/historical_annual.py`
- Utilities in `simulations/utils.py`
- Customization via Streamlit UI in `spectre_web_app.py`
- Supports both 12-row and (future) 48-row historical monthly inputs

### Requirements
- Python 3.8+
- `pandas`, `numpy`, `streamlit`, `plotly`, `pyarrow`
