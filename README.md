# DEAttention

Code accompanying the JMMR manuscript **DEAttention: A Contrastive Differential Attention Mechanism for Sentiment Analysis** (MS ID: JMMR-2606-1990).

Public repository: <https://github.com/ganjimostafa/DE-attention>

## What is included

| Path | Description |
|------|-------------|
| `config.yaml` | Hyperparameters, seeds, split protocol |
| `notebooks/pure_models.ipynb` | Colab / local runner for multi-seed **main**, **ablation**, and **m_sweep** |
| `scripts/` | Notebook builder + result merge / LaTeX helpers |
| `results/` | Per-run JSON checkpoints and summary CSVs used in the revision tables |

## Datasets (not bundled)

Download CSVs yourself and point `config.yaml` / notebook `CONFIG["data_paths"]` at them:

- **IMDb** (50k): [LeenSMN/IMDB-50k-reviews](https://huggingface.co/datasets/LeenSMN/IMDB-50k-reviews) (`review`, `sentiment`)
- **Twitter / Sentiment140**: [kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) — experiments use a binary `text`/`label` CSV with 1,360,000 rows
- **Yelp Review Full** (optional / deferred multi-seed): [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) — binary map drops 3★; 1–2→neg, 4–5→pos

## Protocol (matches the revised manuscript)

- IMDb: fixed contiguous 25k/25k train/test of the released file; 10% of train → validation
- Twitter: stratified 20k test + ≤200k train subsample (seeded); 10% of train → validation
- Early stopping on **validation** Acc; **test evaluated once** at the best-val checkpoint
- Vocabulary built from the **training** split only
- Seeds: IMDb main/ablation `0..4`; Twitter main/ablation/m-sweep and IMDb m-sweep `0..9`

## Quick start (Colab)

1. Open `notebooks/pure_models.ipynb` in Google Colab (T4 GPU recommended).
2. Mount Drive and set `CONFIG["data_paths"]` / `CONFIG["output_dir"]` (or mirror `config.yaml`).
3. Run sessions one mode at a time: `main` → `ablation` → `m_sweep`.
4. Completed runs are skipped if the JSON checkpoint already exists in `output_dir`.

## Local notes

```bash
pip install -r requirements.txt
# Place CSVs under ./data/ and edit config.yaml paths
# Prefer running the notebook; scripts/build_phase3_notebook.py regenerates it
```

## Models

- `standard` — Transformer encoder baseline  
- `diff` — Differential Transformer-style baseline  
- `de_full` — DEAttention: \(S_1 + m(S_2 - S_3)\) with shared \(V\)  
- `de_best` / `jde` — DE/best/1/bin and self-adaptive jDE variants  
- Ablations: `de_avg3`, `de_learned`, `de_sum`, `de_nobase`, `de_indepV`

## Citation

Please cite the JMMR paper (title above) when using this code.

## License

Research code released for reproducibility of the manuscript experiments.
