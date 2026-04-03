# Evidence-Bound Ontology State Engine

Stateful graph updates with **learned relation normalization** (model label -> canonical relation via data-derived mapping), **entity canonicalization**, **per-note unsupported** metrics, **structured conflict records**, and **aligned QA** generation.

## Workflow

1. **QA (recommended):** build triple-consistent questions/golds that match the graph schema:

```bash
python3 state_engine/generate_aligned_qa.py \
  --paraphrase_sets baselines/data/paraphrases/paraphrase_sets_50.json \
  --output results/qa_aligned_100.json \
  --qa_count 100
```

2. **Run the engine** on extractor JSON (e.g. `results/paraphrase_results.json`):

```bash
python3 state_engine/run_state_engine.py \
  --paraphrase_results results/paraphrase_results.json \
  --source_baseline baseline_3 \
  --qa_file results/qa_aligned_100.json \
  --drift_tau 0.2 \
  --output_json results/state_engine_results.json \
  --output_md results/state_engine_results.md
```

Optional UMLS-backed strings (same helper as baselines, no edits under `baselines/`):

```bash
python3 state_engine/run_state_engine.py \
  --mapper_file baselines/data/cui_map.txt
```

**LLM graph QA (retrieve triples → model answers):** default QA in JSON is **symbolic** (`baselines/qa_eval.py`: index lookup on `(subject, relation)` / `(relation, object)`). To have a **local HF model** read retrieved graph lines and answer the natural-language question, add `--llm_graph_qa` (uses the same Qwen stack as baseline 3; needs `torch` + `transformers`). Metrics and per-question rows are written under `qa_llm_graph_rag` in `state_engine_results.json`.

```bash
python3 state_engine/run_state_engine.py \
  --paraphrase_results results/paraphrase_results.json \
  --qa_file results/qa_aligned_100.json \
  --llm_graph_qa \
  --llm_graph_qa_model Qwen/Qwen2.5-0.5B
```

## Outputs

- **`results/state_engine_results.json`** — per-set GED/Jaccard, `Drift@tau`, per-note unsupported, `conflict_records` on edges, QA metrics, slices (top drift sets, QA misses with missing canonical edges).
- **`results/state_engine_results.md`** — summary table + error slices.

### Tables + graph (HTML / PNG)

```bash
python3 state_engine/visualize_results.py \
  --json results/state_engine_results.json \
  --html_out results/state_engine_report.html \
  --graph_png results/state_engine_graph_set0.png \
  --set_id 0
```

Open `state_engine_report.html` in a browser. Optional: `pip install networkx` for spring-layout graphs; without it, a simple circular layout is used.

## Module layout

| File | Role |
|------|------|
| `align.py` | Map extractor triples → canonical relation + oriented `(s,o)` + entity normalization |
| `relations.py` | Loaded label map + relation normalization + canonical relation set |
| `canonicalize.py` | Lowercase/punct, light plural, synonym map; optional mapper hook |
| `ontology.py` | `OntologyAligner` wrapping mapper + `canonicalize` |
| `engine.py` | Incremental state, provenance, `conflict_type` / `source_ids` / `resolution_status` |
| `metrics.py` | GED/Jaccard, `unsupported_vs_note_evidence`, `drift_rate_at_tau` |
| `generate_aligned_qa.py` | Emit schema-aligned QA JSON |
| `llm_graph_qa.py` | Optional LLM QA: lexical triple retrieval + Qwen reads graph |
| `run_state_engine.py` | End-to-end eval + reporting |
