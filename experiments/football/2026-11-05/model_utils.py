"""Features shared by training and inference. No current-match outcomes are inputs."""
import pandas as pd
NUMERIC=[f'{side}_{suffix}' for side in ['home','away'] for suffix in ['prior_matches','prior_points_per_game','prior_goals_for_avg','prior_goals_against_avg','last5_points_per_game']]
DIFFS=['prior_points_per_game','prior_goals_for_avg','prior_goals_against_avg','last5_points_per_game']
FEATURES=NUMERIC+['diff_'+s for s in DIFFS]+['league']
def prepare(frame):
    x=frame[NUMERIC+['league']].copy()
    for c in NUMERIC:x[c]=pd.to_numeric(x[c],errors='coerce')
    for s in DIFFS:x['diff_'+s]=x['home_'+s]-x['away_'+s]
    return x[FEATURES]

def historical_features(matches,league,season,home,away,day):
    past=matches[(matches.league==league)&(matches.season==season)&(matches.date<day)].sort_values(['date','match_id'])
    result={'league':league}
    for side,team in [('home',home),('away',away)]:
        games=past[(past.home_team==team)|(past.away_team==team)]
        points=[];gf=[];ga=[]
        for g in games.itertuples():
            a,b=(g.home_goals,g.away_goals) if g.home_team==team else (g.away_goals,g.home_goals)
            gf.append(a);ga.append(b);points.append(3 if a>b else 1 if a==b else 0)
        n=len(points)
        result.update({side+'_prior_matches':n,side+'_prior_points_per_game':sum(points)/n if n else None,side+'_prior_goals_for_avg':sum(gf)/n if n else None,side+'_prior_goals_against_avg':sum(ga)/n if n else None,side+'_last5_points_per_game':sum(points[-5:])/len(points[-5:]) if n else None})
    return result
