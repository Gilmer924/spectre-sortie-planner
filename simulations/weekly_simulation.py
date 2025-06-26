# -*- coding: utf-8 -*-
"""
Weekly Simulation Engine for SPECTRE
Includes realistic break/fix carryover logic and GA fix integration
"""

import re
import numpy as np
from simulations.simulation_base import SimulationBase

class WeeklySimulation(SimulationBase):
    def validate_params(self):
        weeks = self.params.get('weeks')
        if not isinstance(weeks, int) or weeks < 1:
            raise ValueError("Missing or invalid 'weeks' (int ≥ 1)")

        pats = self.params.get('turn_patterns')
        if not isinstance(pats, list):
            raise ValueError("turn_patterns must be a list")

        weeks_list = pats if isinstance(pats[0], list) else [pats]
        if len(weeks_list) != weeks:
            raise ValueError(f"Expected {weeks} weeks of patterns, got {len(weeks_list)}")

        for wk in weeks_list:
            if not isinstance(wk, list) or len(wk) != 7:
                raise ValueError("Each week must be a list of 7 day-patterns")
            for day_pat in wk:
                if not re.findall(r'\d+', day_pat):
                    raise ValueError(f"Invalid day pattern '{day_pat}'")

        required = [
            'TAI','Overhead','Break %','GA %',
            'Fix-8 Hr','Fix-12 Hr','Fix-24 Hr',
            'Wx Attr %','Sortie Attr %'
        ]
        for key in required:
            val = self.params.get(key)
            if val is None:
                raise ValueError(f"Missing parameter '{key}'")
            try:
                float(val)
            except:
                raise ValueError(f"Parameter '{key}' must be numeric")

    def simulate(self, trials=100, weekend_duty=None):
        self.validate_params()
        TAI = float(self.params['TAI'])
        overhead = float(self.params['Overhead'])
        break_rate = float(self.params['Break %']) / 100.0
        ga_rate = float(self.params['GA %']) / 100.0
        fix8_rate = float(self.params['Fix-8 Hr']) / 100.0
        fix12_rate = float(self.params['Fix-12 Hr']) / 100.0
        fix24_rate = float(self.params['Fix-24 Hr']) / 100.0
        wx_attr = float(self.params['Wx Attr %']) / 100.0
        sortie_attr = float(self.params['Sortie Attr %']) / 100.0

        raw = self.params['turn_patterns']
        weeks = self.params['weeks']
        if not isinstance(raw[0], list):
            raw = [raw]
        patterns = [
            [list(map(int, re.findall(r'\d+', p))) for p in wk]
            for wk in raw
        ]

        if weekend_duty is None:
            weekend_duty = [True] * weeks

        all_trials = []
        for _ in range(trials):
            fix_queues = []
            trial_weeks = []

            for w_idx, week_pats in enumerate(patterns, start=1):
                # Week initialization
                if w_idx == 1:
                    available = TAI - overhead
                else:
                    available = trial_weeks[-1]['ending_available']

                total_sched = total_flown = total_break = 0
                total_weather = total_ga = total_attr = total_mnd = 0
                fix_assigned = fix_completed = 0
                fix_counts = {'8hr':0,'12hr':0,'24hr':0,'long':0}

                daily_avail_start = []
                daily_avail_end = []
                daily_losses = []
                daily_sched = []

                for day_idx, goes in enumerate(week_pats):
                    # Record start-of-day availability
                    daily_avail_start.append(available)

                    # Repairs (only on Mon-Fri or if weekend duty allowed)
                    if weekend_duty[w_idx-1] or day_idx < 5:
                        completed_today = 0
                        active_jobs = fix_queues
                        fix_queues = []
                        for job in active_jobs:
                            job['remaining'] -= 24
                            if job['remaining'] <= 0:
                                available += 1
                                completed_today += 1
                            else:
                                fix_queues.append(job)
                        fix_completed += completed_today

                    # Daily scheduled sorties
                    planned_for_day = sum(goes)
                    daily_sched.append(planned_for_day)

                    # Loss tracking
                    losses = {k:0 for k in ['weather','ground_aborts','sortie_attr','mnd','breaks']}

                    for planned in goes:
                        total_sched += planned

                        # Weather cancellations
                        lost_w = planned - np.random.binomial(planned, 1 - wx_attr)
                        losses['weather'] += lost_w; total_weather += lost_w
                        net1 = planned - lost_w

                        # Ground aborts
                        ga = np.random.binomial(net1, ga_rate)
                        losses['ground_aborts'] += ga; total_ga += ga
                        net2 = net1 - ga

                        # Sortie attrition
                        lost_sa = net2 - np.random.binomial(net2, 1 - sortie_attr)
                        losses['sortie_attr'] += lost_sa; total_attr += lost_sa
                        net3 = net2 - lost_sa

                        # Maintenance non-delivery
                        mnd = max(0, net3 - available)
                        losses['mnd'] += mnd; total_mnd += mnd
                        flew = net3 - mnd
                        total_flown += flew

                        # Aircraft break logic
                        br = np.random.binomial(flew, break_rate)
                        losses['breaks'] += br; total_break += br

                        # Assign breaks into fix queues
                        for _ in range(br):
                            r = np.random.rand()
                            if r < fix8_rate:
                                key, ttl = '8hr', 8
                            elif r < fix8_rate + fix12_rate:
                                key, ttl = '12hr', 12
                            elif r < fix8_rate + fix12_rate + fix24_rate:
                                key, ttl = '24hr', 24
                            else:
                                key, ttl = 'long', np.random.uniform(24, 96)
                            fix_counts[key] += 1
                            fix_assigned += 1
                            available -= 1
                            # Prevent negative availability
                            available = max(available, 0)
                            fix_queues.append({'remaining': ttl})

                    # Clamp availability once more before day end
                    available = max(available, 0)

                    # Save end-of-day availability and losses
                    daily_losses.append(losses)
                    daily_avail_end.append(available)

                avg_daily = sum(daily_avail_end) / len(daily_avail_end)
                trial_weeks.append({
                    'week': w_idx,
                    'scheduled': total_sched,
                    'flown': total_flown,
                    'breaks': total_break,
                    'weather': total_weather,
                    'ground_aborts': total_ga,
                    'sortie_attr': total_attr,
                    'mnd': total_mnd,
                    'fix_assigned': fix_assigned,
                    'fix_completed': fix_completed,
                    **fix_counts,
                    'daily_available_start': daily_avail_start,
                    'daily_available_end': daily_avail_end,
                    'daily_schedule': daily_sched,
                    'daily_losses': daily_losses,
                    'avg_daily_avail': avg_daily,
                    'ending_available': available
                })

            all_trials.append(trial_weeks)
        return all_trials

    def run(self):
        single = self.simulate(trials=1)[0]
        self.results = single
        return single
