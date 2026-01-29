#!/bin/bash

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
export PROJ_ROOT=$(pwd)

# Model Paths (Adjust these if yours are elsewhere)
export HF_MODELS="$HOME/hf_models"
export BASE_MODEL="$HF_MODELS/Qwen2.5-VL-3B-Instruct"
export ST_MODEL="$HF_MODELS/all_MiniLM-L6-v2"


export ANNOT_DIR="$PROJ_ROOT/src/annotations"
export DATA_DIR="$PROJ_ROOT/src/data"
export TRAIN_DIR="$PROJ_ROOT/train_VL"
export EVAL_DIR="$PROJ_ROOT/eval"
export RESULTS_DIR="$EVAL_DIR/results"

mkdir -p "$RESULTS_DIR"
mkdir -p "$EVAL_DIR/json_insample"
mkdir -p "$EVAL_DIR/json_outsample"

# module purge
# module load python/python-3.11.4
echo "Pipeline starting from: $PROJ_ROOT"
echo "Using Python: $(python --version)"

# Every step's results are saved in the repo so evaluation can be run without having to run training again

# # ==============================================================================
# # 1. GENERATE DATA
# # ==============================================================================
# cd "$DATA_DIR"
# JOB1=$(sbatch --parsable ./generate.slurm)
# 
# cd "$PROJ_ROOT/all_data/UCR_dataset/"
# # This job waits for JOB1 to finish successfully
# JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 ucr_data_prep.slurm)
#
#

# # ==============================================================================
# # 2. GENERATE ANNOTATIONS & BASELINES
# # ==============================================================================
#cd "$PROJ_ROOT"

# # Annotations
# python "$ANNOT_DIR/gpt_annotate.py" --image-mode ts1 --base-name annotations --filter
# python "$ANNOT_DIR/gpt_annotate.py" --image-mode ts2 --base-name annotations --filter
# python "$ANNOT_DIR/gpt_annotate.py" --image-mode ts3 --base-name annotations --filter

# # Baselines
# python "$ANNOT_DIR/gpt_baseline.py" --base-name "$EVAL_DIR/baselines/baseline-4o-insample"
# python "$ANNOT_DIR/gpt_baseline.py" --base-name "$EVAL_DIR/baselines/baseline-gpt5.2-insample"
# python "$ANNOT_DIR/gpt_baseline_outsample.py" --base-name "$EVAL_DIR/baselines/baseline-4o-outsample"
# python "$ANNOT_DIR/gpt_baseline_outsample.py" --base-name "$EVAL_DIR/baselines/baseline-gpt5.2-outsample"

# 
# # ==============================================================================
# # 3. TRAIN QWEN-VL
# # ==============================================================================
# cd "$TRAIN_DIR"
# # This job waits for the data prep (JOB2) to finish
# # JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 train_clip.slurm)
# JOB3=$(sbatch --parsable train_clip.slurm)

# # ==============================================================================
# # 4. EVALUATION
# # ==============================================================================
# cd "$EVAL_DIR/scripts/"
# # These two can run in parallel, but both depend on Training (JOB3)
# JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 eval_outsample.slurm)
# JOB5=$(sbatch --parsable --dependency=afterok:$JOB3 eval_insample.slurm)
#
# # # ==============================================================================
# # 5. AGGREGATION & SUMMARIZATION
# # ==============================================================================
#
# echo "Waiting for evaluation jobs ($JOB4, $JOB5) to complete before aggregating..."
#
# wait_for_jobs() {
#     while squeue -j $(echo "$@" | tr ' ' ',') -h | grep -q ^; do
#         sleep 30
#     done
# }
#
# srun --dependency=afterok:$JOB4:$JOB5 --jobid=$JOB4 sleep 0 2>/dev/null || wait_for_jobs $JOB4 $JOB5
#
# # Affiliation metrics:
cd "$EVAL_DIR/scripts/"
python aggregate_affil.py  \
    --out "$RESULTS_DIR/outsample_affil_summary.csv" \
    "$EVAL_DIR/baselines/baseline-park-outsample.jsonl" \
    "$EVAL_DIR/baselines/baseline-4o-outsample_ts1.jsonl" \
    "$EVAL_DIR/baselines/baseline-gpt5.2-outsample_ts1.jsonl" \
    "$EVAL_DIR/json_outsample"/*.jsonl

python aggregate_affil.py  \
    --out "$RESULTS_DIR/insample_affil_summary.csv" \
    --synth \
    "$EVAL_DIR/baselines/baseline-park-insample.jsonl" \
    "$EVAL_DIR/baselines/baseline-4o-insample_ts1.jsonl" \
    "$EVAL_DIR/baselines/baseline-gpt5.2-insample_ts1.jsonl" \
    "$EVAL_DIR/json_insample"/*.jsonl

 ## Explanation quality:
 ## --- In-Sample Analysis ---
python citation_summary.py \
    --out-citations "$RESULTS_DIR/insample_expl_summary.csv" \
    --out-explanations "$RESULTS_DIR/insample_error_detail_summary.csv" \
    --answer-key output \
    --synth \
    "$EVAL_DIR/baselines/baseline-4o-insample_ts1.jsonl" \
    "$EVAL_DIR/baselines/baseline-gpt5.2-insample_ts1.jsonl" \
    "$EVAL_DIR/json_insample"/*.jsonl \
    --plain-text

 ## --- Out-of-Sample Analysis ---
python citation_summary.py \
    --out-metrics "$RESULTS_DIR/outsample_metrics_summary.csv" \
    --answer-key output \
    "$EVAL_DIR/baselines/baseline-4o-outsample_ts1.jsonl" \
    "$EVAL_DIR/baselines/baseline-gpt5.2-outsample_ts1.jsonl" \
    "$EVAL_DIR/json_outsample"/*.jsonl \
    --plain-text

echo "Pipeline sequence completed."

