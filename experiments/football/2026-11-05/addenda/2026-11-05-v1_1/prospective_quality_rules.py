"""Pre-outcome addendum: deterministic odds and coverage rules, v1.1.
This module does NOT alter locked forecast models or the original protocol.
"""
import math
from datetime import datetime,timezone
COHORT_N=1000
ADMIN_CUTOFF=datetime(2027,6,30,23,59,59,tzinfo=timezone.utc)

def closing_eligibility(odds):
    if not isinstance(odds,(list,tuple)) or len(odds)!=3:
        return {'eligible':False,'reason':'missing_triplet','probabilities_ADH':None}
    try:values=[float(x) for x in odds]
    except (ValueError,TypeError):return {'eligible':False,'reason':'non_numeric','probabilities_ADH':None}
    if not all(math.isfinite(v) and v>1 for v in values):
        return {'eligible':False,'reason':'nonfinite_or_odds_le_one','probabilities_ADH':None}
    implied=[1/v for v in values];total=sum(implied)
    if total<1-1e-12 or total>1.20+1e-12:
        return {'eligible':False,'reason':'outside_registered_overround_band','sum_inverse_odds':total,'probabilities_ADH':None}
    return {'eligible':True,'reason':'eligible','sum_inverse_odds':total,'probabilities_ADH':[v/total for v in implied]}

def coverage_status(completed_by_cutoff,verified_forecasts,as_of,analysis_already_run=False):
    if not all(isinstance(n,int) and 0<=n<=COHORT_N for n in [completed_by_cutoff,verified_forecasts]):
        raise ValueError('Counts must refer to the fixed 1000 fixture identities.')
    if as_of.tzinfo is None:raise ValueError('UTC-aware timestamp required')
    if analysis_already_run:return 'already_analysed_no_repeat'
    if completed_by_cutoff==COHORT_N and verified_forecasts==COHORT_N:return 'ready_for_single_pilot_analysis'
    if as_of>=ADMIN_CUTOFF:return 'closed_incomplete_no_confirmatory_claim'
    return 'waiting_report_counts_only'

def market_coverage(valid_closing_rows):
    if not isinstance(valid_closing_rows,int) or not 0<=valid_closing_rows<=COHORT_N:raise ValueError('Invalid count')
    return {'denominator':COHORT_N,'numerator':valid_closing_rows,'fraction':valid_closing_rows/COHORT_N,'passes_95pct':valid_closing_rows>=950}
