# Football prospective registration — 2026-11-05

This commit publishes the forecast files and their SHA-256 before the listed matches.
No prior historical test is claimed to be untouched; all historical comparisons are diagnostic.

## Existing static-feature pilot: 144 matches

`pilot144_forecasts.csv` and `pilot144_manifest.json` are unchanged copies of the
Nov 5 snapshot for Nov 6–30 fixtures. Models and features are frozen through Nov 2.
This is a separate static-feature pilot, not a daily-updating evaluation.

## Rolling cohort: exactly 1000 matches

`rolling1000_cohort.csv` is a fixed cohort selected before outcomes, scheduled
Nov 6, 2026 through Apr 17, 2027 in the pinned fixture snapshot. No replacement
fixtures may be added to obtain significance. The existing boosting production
model is primary; the fixed logistic-since-2010 model is the comparator.
Weights, preprocessing and calibration stay fixed; future past-result features
are updated by the preregistered daily rule.

The primary criterion: paired mean log loss (boosting minus logistic), monthly
cluster bootstrap, 20000 resamples, seed=0, percentile 95% CI. Boosting counts as
better ONLY if the upper CI endpoint is strictly below zero. Full protocol,
missing-data rules, market benchmark and power caveats are in
`rolling1000_protocol.json`. Final analysis happens only after the fixed cohort
has outcomes and admissible publicly recorded pre-match forecasts. No optional
stopping or repeated significance decisions. A 1000-match cohort is a pilot,
not an 80%-power design for a 0.007 effect at the measured historical SD.

Only the first day (five fixtures on Nov 6) has rolling forecasts at registration.
Future batches require daily execution and public recording before kickoff.
NO background scheduler or automatic future publishing is configured. Local hashes
alone are insufficient for later batches. Missing publication means incomplete
coverage; it must not be silently replaced with later fixtures.

The next-day batch must be generated before 12:00 UTC on the prior day, using only
completed results dated strictly before that generation day. Forecasts for postponed
games retain the first admissible publicly recorded version. See the protocol for
rescheduling and missing-result rules. Independent public push/receipt timing is
needed in addition to Git author dates, which can be set manually.

Model hashes are in the manifests. Model artifacts are in the separately supplied
model package; this registration contains no secret values. Forecast probabilities
are analytics, not advice. Closing-price comparisons do not imply profitability.
