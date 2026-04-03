# Baselines

This folder holds three pipelines for turning biomedical text into knowledge-graph triples (subject, relation, object). Each one is “stateless”: you pass in text and get back triples for that text only. There’s no persistent graph—every call overwrites the previous result.

**Baseline 1** uses a standard NER + relation setup: BioBERT-based token classification for entities, then a relation model over entity pairs. Output is surface text (no normalization).

**Baseline 2** is the same stack as 1, but entities are mapped to MeSH/CUI when possible. The mapper is built from the BC5CDR corpus if you have it, or you can point to a file (`data/cui_map.txt` or `--mapper_file`) with lines like `entity_text<TAB>CUI`. So you get triples in a more canonical form.

**Baseline 3** skips NER and relation models entirely. It uses Qwen 2.5 with a fixed prompt to extract triples directly from the sentence. No training, no ontology—just the language model.

Evaluation runs all three by default: incremental simulation (random orderings of sentences) and paraphrase evaluation (same fact, different phrasings). Metrics are graph edit distance, Jaccard similarity, unsupported inference rate, and a simple conflict heuristic. There’s also a downstream QA step that answers questions over the extracted graph.

---

## Getting started

From the project root, go into `baselines` and install dependencies:

```bash
cd baselines
pip install -r requirements.txt
```

Baselines 1 and 2 need a trained relation model (and optionally NER). If you only want to use a pretrained NER from the Hub, you can skip NER training and just train the relation head:

```bash
python train.py --dataset bc5cdr --model_name dmis-lab/biobert-base-cased-v1.1 --skip_ner_training
```

Checkpoints are written to `out/ner` and `out/relation`. To train NER as well, drop the `--skip_ner_training` flag. Baseline 3 doesn’t use these at all—it only needs the Qwen model.

To run the full evaluation (incremental + paraphrase, all three baselines), no arguments are required:

```bash
python run_graph_eval.py
```

Results land in `results/`: `eval_results.json` for the incremental run, `paraphrase_results.json` when a paraphrase file is present (e.g. `data/paraphrases/paraphrase_sets_5.json` or `paraphrase_sets_30.json`). Plots go to `results/plots/`. Each JSON has entries for `baseline_1`, `baseline_2`, and `baseline_3` with summary stats (mean, 95% CI) and the raw metric arrays.

To run QA over the extracted graph after that:

```bash
python run_qa_eval.py
```

This reads from `results/paraphrase_results.json` by default (or you can pass `--graph_file` for a single graph). It prints Exact Match and Recall@1 and writes `results/qa_results.json` and `results/qa_results.txt`.

---

## What each script does

- **`train.py`** — Trains NER (MedMentions or BC5CDR) and/or the relation model (BC5CDR). Handles sentence-level splitting and optional CRF for NER. You can train only the relation part and use a HuggingFace NER model with `--skip_ner_training` and `--ner_pretrained_model`.

- **`run_baseline.py`** — Run a single baseline on one sentence, e.g. `python run_baseline.py 1` or `2` or `3`. Handy for quick checks. Use `--text "Your sentence."` to override the default example. For baseline 2 you can pass `--mapper_file`; for baseline 3, `--qwen_model`. The `--diagnose` flag prints NER spans and relation predictions for 1 and 2.

- **`run_graph_eval.py`** — Runs the incremental and paraphrase experiments, writes JSON and (unless you pass `--no_plots`) the plots. You can restrict baselines with `--baselines 1,3` or point to another paraphrase file with `--paraphrases_file`.

- **`run_qa_eval.py`** — Builds a graph index from the triples, runs the QA examples (subject/relation/object with `?` for the slot to fill), and reports Exact Match and Recall@1. For baseline 2 it uses the same entity→ID mapping so answers are compared in canonical form.

---

## Folder structure

- **`loaders/`** — Data loading for MedMentions and BC5CDR (PubTator format), plus sentence splitting.
- **`models/`** — BERT NER (`bert_ner.py`) and BERT relation (`bert_relation.py`). Both support loading from a local checkpoint or (for NER) from the Hub.
- **`pipelines/`** — The three baselines: `stateless_pipeline.py` (1), `ontology_stateless_pipeline.py` (2), `single_extractor_pipeline.py` (3). Each exposes a `process(text)` that returns a list of triples.
- **`ontology/`** — `umls_mapper.py` for mapping entity strings to CUI/MeSH (from file, BC5CDR annotations, or MedMentions).
- **`extractors/`** — `qwen_prompt_extractor.py` for baseline 3: prompt-based triple extraction with Qwen 2.5.
- **`drift_metrics.py`** — Graph edit distance, Jaccard on triple sets, unsupported-inference rate, and a heuristic for relation conflicts (same subject–object, different relation).
- **`qa_eval.py`** — Builds the graph index, runs queries (subject→relation→? or ?→relation→object), and computes QA metrics.
- **`experiment_runner.py`** — Incremental simulation (random orderings), paraphrase pairwise metrics, and summary statistics (mean, 95% CI).
- **`graph_eval_plots.py`** — Plotting for incremental and paraphrase results (e.g. distributions, heatmaps, first-five-paraphrase figures).

---

## Data you’ll need

- **MedMentions** — Used for NER training if you choose that dataset. The loader can download it if `data/medmentions/` is empty.

- **BC5CDR** — For NER and relation training. Put the PubTator-style `.txt` files in `data/bc5cdr/`. The relation trainer expects document-level data; sentence splitting is done inside the loader. If you have BC5CDR in place, baseline 2 will use the corpus to build the entity→MeSH mapper unless you pass `--no_corpus_mapper`.

- **Paraphrases** — JSON with either a single list under `paraphrases` (and optional `fact`) or a list of sets under `paraphrase_sets` (each with `fact` and `paraphrases`). The repo uses `data/paraphrases/paraphrase_sets_5.json` or `paraphrase_sets_30.json` when present; otherwise it falls back to something like `aspirin_fact.json`.

- **QA** — JSON in `data/qa/` with an `examples` array (each item: question, subject, relation, object, answer). Use `?` for the slot you want to retrieve. Optional `relation_aliases` map canonical relation names to the strings that appear in the graph (e.g. for LABEL_0-style relations). Default QA file is `data/qa/qa_50.json` or `aspirin_qa.json` if that’s what exists.

If you generated the 30 paraphrase sets and 50 QA examples with the data script, `run_graph_eval.py` and `run_qa_eval.py` will pick those up automatically when run with no extra flags.
