"""Simulation audit of the registered percentile month-bootstrap rule.
No future match labels and no changes to the registered models or forecasts.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
from pathlib import Path
import json,math
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'audit_simulation';O.mkdir(exist_ok=True)
cohort=pd.read_csv(R/'prospective_rolling_1000_v1/cohort.csv');month_sizes=cohort.groupby(cohort.scheduled_date.str[:7]).size();sizes=month_sizes.to_numpy();G=len(sizes);N=int(sizes.sum())
obs=pd.read_csv(R/'audit_bootstrap/logistic_since2010_test_predictions.csv');differences=obs.loss_difference_boosting_minus_logistic.to_numpy();sd=float(differences.std(ddof=1));M=5000;B=20000
# Use precisely the registered seed and number of resamples. Collapse identical
# block multiplicity vectors: for 6 blocks there are only C(11,5)=462 possibilities.
rng=np.random.default_rng(0);sample=rng.integers(0,G,size=(B,G));multiplicity=np.column_stack([(sample==i).sum(axis=1) for i in range(G)]);vectors,frequency=np.unique(multiplicity,axis=0,return_counts=True)
weights=vectors*sizes[None,:];weights=weights/weights.sum(axis=1,keepdims=True)

def quantile_from_counts(values,frequency,q):
    # values shape (unique bootstrap configurations, outer simulations).
    order=np.argsort(values,axis=0);sorted_values=np.take_along_axis(values,order,axis=0)
    cumulative=np.cumsum(frequency[order],axis=0)
    position=(B-1)*q;lo=int(np.floor(position));hi=int(np.ceil(position));frac=position-lo
    i=np.argmax(cumulative>lo,axis=0);j=np.argmax(cumulative>hi,axis=0);columns=np.arange(values.shape[1])
    return (1-frac)*sorted_values[i,columns]+frac*sorted_values[j,columns]

def bootstrap_bounds(means):
    low=[];high=[]
    for offset in range(0,len(means),250):
        values=weights@means[offset:offset+250].T
        low.extend(quantile_from_counts(values,frequency,.025));high.extend(quantile_from_counts(values,frequency,.975))
    return np.array(low),np.array(high)

def wilson(k,n):
    z=1.959963984540054;v=k/n;den=1+z*z/n;mid=(v+z*z/(2*n))/den;half=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def normal_means(seed,rho,phi=0):
    rng=np.random.default_rng(seed);u=rng.normal(size=(M,G))
    if phi:
        for j in range(1,G):u[:,j]=phi*u[:,j-1]+np.sqrt(1-phi*phi)*u[:,j]
    return sd*np.sqrt(rho)*u+sd*np.sqrt((1-rho)/sizes)[None,:]*rng.normal(size=(M,G))

scenarios={
 'normal_iid':(normal_means(20261105,0),'Independent normal match differences; no month effect.'),
 'normal_month_icc_0005':(normal_means(20261106,.005),'Normal shared month intercept; within-month ICC=0.005, independent months.'),
 'normal_month_icc_001':(normal_means(20261107,.01),'Normal shared month intercept; within-month ICC=0.01, independent months.'),
 'normal_month_icc_005':(normal_means(20261108,.05),'Normal shared month intercept; within-month ICC=0.05, independent months.'),
 'normal_month_icc_001_ar05':(normal_means(20261109,.01,.5),'Normal shared month intercept; ICC=0.01; AR(1) correlation 0.5 across adjacent months.')}
# Empirical iid: center the observed paired differences under a zero-mean null.
rng=np.random.default_rng(20261110);centered=differences-differences.mean();draws=centered[rng.integers(0,len(centered),size=(M,N))];starts=np.r_[0,np.cumsum(sizes)]
means=np.column_stack([draws[:,starts[j]:starts[j+1]].mean(axis=1) for j in range(G)])
scenarios['empirical_iid']=(means,'Independent resampling of centered observed match-level differences; heavy tails retained, dependencies removed.')
del draws
# Empirical month scenario: uniformly sample historical months, center the
# unweighted month means so the synthetic null is exactly zero in expectation.
months=obs.date.str[:7];pools=[differences[months==name] for name in sorted(months.unique())];poolmeans=np.array([p.mean() for p in pools]);month_effect=poolmeans-poolmeans.mean();residuals=[p-p.mean() for p in pools]
rng=np.random.default_rng(20261111);source_month=rng.integers(0,len(pools),size=(M,G));empirical=np.empty((M,G))
for j,n in enumerate(sizes):
    for k,pool in enumerate(residuals):
        which=np.flatnonzero(source_month[:,j]==k)
        if not len(which):continue
        sample=pool[rng.integers(0,len(pool),size=(len(which),int(n)))].mean(axis=1)
        empirical[which,j]=month_effect[k]+sample
scenarios['empirical_month_resample']=(empirical,'Synthetic centered historical-month mixture: sample historical months uniformly, resample centered within-month residuals, retain month-mean heterogeneity. It is one assumed data-generating process, not proof of future dependence.')
results=[]
for name,(means,description) in scenarios.items():
    lower,upper=bootstrap_bounds(means)
    false=int((upper<0).sum());coverage=int(((lower<=0)&(upper>=0)).sum());effect=-.007
    # Adding a fixed effect shifts each bootstrap quantile by that effect.
    power=int((upper+effect<0).sum())
    row=dict(scenario=name,description=description,outer_simulations=M,inner_resamples=B,blocks=G,n=N,false_confirmation_rate=false/M,false_confirmation_mc_ci95=wilson(false,M),null_ci_coverage=coverage/M,coverage_mc_ci95=wilson(coverage,M),power_at_delta_minus_0007=power/M,power_mc_ci95=wilson(power,M),probability_no_confirmation_at_minus_0007=1-power/M)
    results.append(row);print(json.dumps(row),flush=True)
# Verify the compressed bootstrap computation against the literal 20,000 values.
test=np.array([np.arange(G)/100, -np.arange(G)/100, np.full(G,-.007)])
lo,hi=bootstrap_bounds(test)
literal=(multiplicity@(test*sizes).T)/(multiplicity@sizes)[:,None]
np.testing.assert_allclose(lo,np.percentile(literal,2.5,axis=0),atol=1e-14);np.testing.assert_allclose(hi,np.percentile(literal,97.5,axis=0),atol=1e-14)
assert all(0<=r['false_confirmation_rate']<=1 for r in results)
report=dict(registered_rule='Upper endpoint of a TWO-SIDED percentile 95% month-bootstrap CI < 0.',nominal_left_tail_under_valid_interval=.025,note='This is nominal 2.5%, not 5%; an upper one-sided 95% bound would target 5%. A directional CI decision can be a hypothesis test if its coverage is calibrated.',month_sizes=month_sizes.to_dict(),marginal_historical_sd=sd,outer_simulations=M,inner_resamples=B,bootstrap_seed=0,simulation_seeds='20261105 through 20261111',exact_compression_check=True,scenarios=results,limitations=['Simulations assess only specified synthetic data-generating processes; not a universal type-I error guarantee.','Shared month variance and temporal correlations are assumed scenarios, not estimated truths about future matches.','Original cohort has 6 planned calendar months, not the 24 observed historical months; rescheduling can change the eventual block count.','Only six clusters can cause serious undercoverage; registration prevents outcome-dependent changes but does not validate the statistical method.','The original protocol and frozen forecasts are unchanged. No future match outcomes used.'])
(O/'simulation_results.json').write_text(json.dumps(report,indent=2));pd.DataFrame([{k:v for k,v in r.items() if k not in ['description','false_confirmation_mc_ci95','coverage_mc_ci95','power_mc_ci95']} for r in results]).to_csv(O/'simulation_summary.csv',index=False)
print('PASS: compressed simulation matches literal registered bootstrap on reference arrays.',flush=True)
