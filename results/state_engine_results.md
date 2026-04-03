# Evidence-Bound Ontology State Engine Results

## Summary
| GED mean | Jaccard mean | Drift@0.2 | Unsupported (per-note) | Conflict records | QA EM | QA Recall@1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1613 | 0.8735 | 0.1267 | 0.9080 | 174 | 0.6531 | 0.6837 |

## Top 10 high-drift sets (lowest Jaccard stability)
| set_id | fact | Jaccard | GED | Drift@0.2 |
| --- | --- | --- | --- | --- |
| 44 | Doxycycline treats Lyme disease. | 0.1333 | 1.4000 | 0.8667 |
| 36 | Finasteride treats benign prostatic hyperplasia. | 0.2296 | 1.3556 | 0.7778 |
| 6 | Insulin regulates blood glucose. | 0.3333 | 0.8000 | 0.6667 |
| 48 | Clindamycin treats skin infections. | 0.3333 | 0.8000 | 0.6667 |
| 47 | Rivaroxaban prevents stroke in atrial fibrillation. | 0.4667 | 0.6000 | 0.5333 |
| 28 | Gabapentin treats nerve pain. | 0.4889 | 0.5556 | 0.5111 |
| 45 | Esomeprazole treats gastroesophageal reflux disease. | 0.4889 | 0.5556 | 0.5111 |
| 2 | Warfarin increases the risk of bleeding. | 0.8000 | 0.2000 | 0.2000 |
| 5 | Statins lower cholesterol. | 0.8000 | 0.2000 | 0.2000 |
| 13 | Metformin can cause lactic acidosis. | 0.8000 | 0.2000 | 0.2000 |

## Top 10 QA misses (missing canonical edge)
| question | gold | missing_edge (sub,rel,obj) |
| --- | --- | --- |
| What does warfarin increase? | bleeding | ('warfarin', 'increases', 'bleeding') |
| What increases bleeding? | warfarin | ('warfarin', 'increases', 'bleeding') |
| What does statins reduce? | cholesterol | ('statins', 'reduces', 'cholesterol') |
| What reduces cholesterol? | statins | ('statins', 'reduces', 'cholesterol') |
| What causes blood glucose? | insulin | ('insulin', 'causes', 'blood glucose') |
| What does anticoagulants prevent? | blood clots | ('anticoagulants', 'prevents', 'blood clots') |
| What prevents blood clots? | anticoagulants | ('anticoagulants', 'prevents', 'blood clots') |
| What does metformin cause? | lactic acidosis | ('metformin', 'causes', 'lactic acidosis') |
| What causes lactic acidosis? | metformin | ('metformin', 'causes', 'lactic acidosis') |
| What does aspirin prevent? | heart attacks | ('aspirin', 'prevents', 'heart attacks') |

Full results JSON: `/home/stu1/s7/sp7289/Capstone/results/state_engine_results.json`