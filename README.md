# [DistillAnomaly]

This repository contains a reproducible, Slurm-based pipeline for **time series anomaly detection** experiments: **synthetic data generation**, **UCR data preparation**, **VL model training**, **in-sample and out-of-sample evaluation**, and **aggregation** into CSV summaries.

**Main entry point:** `pipeline.sh` (repository root).

> Slurm is used because the project was developed and executed on an HPC environment. All intermediate and final outputs are provided so the most expensive stages do not need to be rerun.

---

## 1. Pipeline overview

The pipeline is organized into the following stages:
(Each step's parameters can be specified in the scripts that are called by the main pipeline.sh)

- **Synthetic data generation** (`src/data/generate.slurm`)  
  Writes datasets under `all_data/synthetic/` (e.g., `freq/`, `noisy-freq/`) with:
  - `data.pkl`
  - `csv_data/`
  - `figs/`

- **UCR data preparation** (`all_data/UCR_dataset/`)  
  A Slurm preparation job is referenced from `pipeline.sh`.

- **Annotation processing** (`src/annotations/`)  
  Produces/normalizes annotation artifacts used by training/evaluation (including explanation-related analysis).
  Some annotation steps require an OpenAI API key (**`OPENAI_API_KEY`**).

- **VL model training** (`train_VL/`, e.g., `train_clip.slurm`)  
  Uses a local Qwen2.5-VL checkpoint configured in `pipeline.sh`.

- **Evaluation** (`eval/scripts/`)  
  Writes JSONL outputs to:
  - `eval/json_outsample/` (out-of-sample / UCR)
  - `eval/json_insample/` (in-sample / synthetic)

- **Aggregation** (`eval/results/`)  
  - `aggregate_affil.py`: affiliation summaries
  - `citation_summary.py`: explanation/citation-quality + metrics summaries

---

## 2. Reference environment

Reference runs were executed on:
- **Slurm:** 22.05.10  
- **OS:** RHEL  
- **GPU:** NVIDIA H100  

---

## 3. Key directories

- `pipeline.sh` — orchestrates Slurm jobs + aggregation
- `src/data/` — synthetic generation (Slurm)
- `src/annotations/` — annotation processing
- `all_data/synthetic/` — generated synthetic datasets
- `all_data/UCR_dataset/` — UCR datasets
- `train_VL/` — training scripts + Slurm jobs
- `eval/` — evaluation scripts, JSONL outputs, baselines, results
  - `eval/json_insample/`
  - `eval/json_outsample/`
  - `eval/baselines/`
  - `eval/results/`

---

## 4. Configuration

Edit configuration directly in `pipeline.sh` :
- model paths (e.g., `BASE_MODEL`, `ST_MODEL`)
- project paths (`TRAIN_DIR`, `EVAL_DIR`, etc.)
- output locations

If you run annotation processing that requires OpenAI access, set:
```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
````


## 5. Run

From the repository root:

```bash
sh pipeline.sh
```

---

## 6. Dependencies

Install Python dependencies via:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

