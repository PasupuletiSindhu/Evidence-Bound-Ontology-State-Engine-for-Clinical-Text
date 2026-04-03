# Method comparison: baselines vs evidence-bound state engine

**Inputs** (paths relative to repository root where applicable)

- Paraphrase / extractor log: `results/paraphrase_results.json`
- QA-over-graph (baselines): `results/qa_results.json`
- State engine: `results/state_engine_results.json`

**Fair QA**: Regenerate QA results with the same `--qa_file` you pass to `run_state_engine.py`, e.g.
`python baselines/run_qa_eval.py --paraphrase_results results/paraphrase_results.json --qa_file results/qa_aligned_100.json --output results/qa_results.json`

**Graph metrics (GED / Jaccard / drift)**: Baseline columns use raw triples from each baseline pipeline. State-engine columns use **canonicalized, oriented** triples before pairwise stability (see `prepare_triples_for_state`). Trend comparison is meaningful; numerical equality across rows is not required.

## Summary table

| Method | GED mean | Jaccard mean | Drift@0.2 | QA EM | QA R@1 | Unsupported (note) | Conflict rec. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline 1 (stateless NER+RE) | 1.1990 | 0.2919 | 0.7212 | 0.0408 | 0.0408 | — | — |
| Baseline 2 (+ ontology IDs) | 0.9273 | 0.4129 | 0.6111 | 0.0408 | 0.0408 | — | — |
| Baseline 3 (Qwen) | 2.0213 | 0.3138 | 0.6907 | 0.4286 | 0.4286 | — | — |
| Evidence-bound state engine (aligned + incremental) | 0.1613 | 0.8735 | 0.1267 | 0.6531 | 0.6837 | 0.9080 | 174 |

Same numbers are in `results/method_comparison.csv` for plotting or spreadsheets.

## Results figure (example graph, set 0)

Merged view for the first paraphrase set: synonyms combined; low-information object nodes omitted for clarity. **Active** edges (dark) are strongly supported; **uncertain** edges (red) need more evidence under the engine's policy.

![Set 0: merged graph — aspirin and gastric outcomes](state_engine_graph_set0.png)

**See also:** `state_engine_report.html` in this folder for the full HTML report.
