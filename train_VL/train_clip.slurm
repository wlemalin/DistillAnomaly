#! /bin/bash

#SBATCH --job-name=ts-train
#SBATCH --output=qwen_%j.out
#SBATCH --error=qwen_%j.err
#SBATCH --gres=gpu:2
#SBATCH --partition=publicgpu
#SBATCH --constraint=gpuh100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=05:00:00

module purge
module load python/python-3.11.4

echo "Running on $(hostname), Python: $(python --version)"
PROJ_DIR=$(pwd)

# Define the model paths here
MODEL_PATH="$HOME/hf_models/Qwen2.5-VL-3B-Instruct"

run_one(){
local data=$1
local out=$2
local epochs=$3

python3 train_txt_vision_clip.py \
    --model_dir "$MODEL_PATH" \
    --data_json "$PROJ_DIR/$data" \
    --out_dir "$PROJ_DIR/$out" \
    --val_ratio 0.10 \
    --seed 42 \
    --epochs $epochs \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-5 \
    --debug \
    --alpha_itc 0.1 \
    --bf16
}

run_one "src/annotations/jsonl/annotations_ts1_filtered.jsonl" \
        "train_VL/qwen2.5-vl-3b-lora-vl-ts1-10ep" \
        10

run_one "src/annotations/jsonl/annotations_ts2_filtered.jsonl" \
        "train_VL/qwen2.5-vl-3b-lora-vl-ts2-15ep" \
        15

run_one "src/annotations/jsonl/annotations_ts3_filtered.jsonl" \
        "train_VL/qwen2.5-vl-3b-lora-vl-ts3-20ep" \
        20

echo "All jobs finished."


