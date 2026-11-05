"""Generate tomorrow's locked rolling forecasts. Never backdates or overwrites."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(n,'2')
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
import pandas as pd,numpy as np,joblib
from model_utils import historical_features,prepare
R=Path(__file__).resolve().parent;D=R/'prospective_rolling_1000_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def generate(results_path):
    now=datetime.now(timezone.utc)
    if now.hour>=12:raise ValueError('Registered deadline is before 12:00 UTC on the day BEFORE fixtures. Wait for the next registered forecast day; never backdate.')
    today=now.date().isoformat();tomorrow=(now.date()+timedelta(days=1)).isoformat()
    protocol=json.loads((D/'protocol.json').read_text())
    for n,h in protocol['locked_file_sha256'].items():
        if sha(D/n)!=h:raise ValueError('Locked input changed: '+n)
    for n,h in protocol['code_sha256'].items():
        if sha(R/n)!=h:raise ValueError('Registered implementation changed: '+n)
    matches=pd.read_csv(results_path)
    if (matches.date>=today).any():raise ValueError('Input contains current/future calendar dates; use only completed results before generation day.')
    if matches.duplicated(['league','season','home_team','away_team']).any():raise ValueError('Ambiguous duplicate results for a league-season pair.')
    cohort=pd.read_csv(D/'cohort.csv');batch=cohort[cohort.scheduled_date==tomorrow].copy()
    if batch.empty:raise ValueError('No registered cohort fixtures tomorrow.')
    target=D/'daily'/tomorrow
    if target.exists():raise FileExistsError('Batch already exists; forecasts must not be replaced.')
    models={name:joblib.load(D/'models'/file)['model'] for name,file in [('boosting','boosting_primary.joblib'),('logistic','logistic_challenger.joblib')]}
    rows=[historical_features(matches,f.league,f.season,f.home_team,f.away_team,tomorrow) for f in batch.itertuples()]
    x=prepare(pd.DataFrame(rows));output=batch.reset_index(drop=True).copy()
    for name,model in models.items():
        assert list(model.classes_)==['A','D','H'];p=model.predict_proba(x);np.testing.assert_allclose(p.sum(axis=1),1)
        for j,c in enumerate(['A','D','H']):output[name+'_probability_'+c]=p[:,j]
    for c in ['A','D','H']:output['baseline_league_probability_'+c]=[protocol['frozen_baseline_probabilities'][league][c] for league in output.league]
    output['generated_at_utc']=now.isoformat();output['data_through']=matches.date.max();output['note']='аналитика, не совет'
    target.mkdir(parents=True,exist_ok=False)
    output.to_csv(target/'forecasts.csv',index=False)
    pd.concat([batch.reset_index(drop=True)[['fixture_id']],x],axis=1).to_csv(target/'features.csv',index=False)
    metadata=dict(experiment_id=protocol['experiment_id'],generated_at_utc=now.isoformat(),forecast_date=tomorrow,n=len(batch),protocol_sha256=sha(D/'protocol.json'),data_sha256=sha(Path(results_path)),data_through=matches.date.max(),feature_cutoff_exclusive=today,forecasts_sha256=sha(target/'forecasts.csv'),features_sha256=sha(target/'features.csv'),publication_status='Local only until public pre-kickoff receipt/commit is recorded. Do not treat local time as independent proof.')
    (target/'batch.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2));return metadata
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
    try:print(json.dumps(generate(a.results),ensure_ascii=False,indent=2))
    except (ValueError,FileExistsError) as e:p.error(str(e))
