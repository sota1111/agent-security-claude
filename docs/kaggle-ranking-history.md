# Kaggle ranking history

- Competition: [AI Agent Security](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks)
- Kaggle team: `sota1111`
- Lineage: Claude

| Observed at (UTC) | Official team rank | Teams | This lineage's observed public score |
| --- | ---: | ---: | ---: |
| 2026-07-26 13:38 | 2,404 | 2,405 | 0.000 |
| 2026-07-28 15:57 | 2,480 | 2,488 | 0.000 |

The official rank is shared by the GPT and Claude repositories because both submit under the same Kaggle team. The score is the latest completed submission identifiable as this lineage at the observation time.

Source: Kaggle CLI `competitions list` and `competitions submissions`. Earlier official-rank snapshots were not retained, so they are not reconstructed or estimated.

## Submission decisions

| Date (UTC) | Issue | Decision | Kaggle ref | Status / score | Reason |
| --- | --- | --- | ---: | --- | --- |
| 2026-07-29 | SOT-2149 | Duplicate skip | 55056847 | Previously pending / score unavailable | Independent-seed confirm did not promote any transfer candidate. The retained champion and embedded kernel payload remain byte-identical to the v3 submission (`e6d75d4b…`), so no duplicate push or submission was warranted. Kaggle credentials were unavailable for a fresh status query. |
| 2026-08-01 | SOT-2222 | Duplicate skip | 55132774 | COMPLETE / 0.000 | Confirm seeds `5,21,44` were disjoint from screen seeds `7,99,123`. Every candidate class and the champion scored target-predicate breach rate `0.000`, so none met the all-seed strict-improvement gate. Champion and embedded payload remain byte-identical (`e6d75d4b…`); Kaggle CLI confirmed the latest Claude-lineage submission already completed at this ref and score, so no duplicate kernel push or submission was made. |
