#! /bin/bash

#SBATCH --job-name=ts-train
#SBATCH --output=qwen_%j.out
#SBATCH --error=qwen_%j.err
#SBATCH --gres=gpu:2
#SBATCH --partition=publicgpu
#SBATCH --constraint=gpuh200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=04:00:00

module purge
module load python/python-3.11.4

echo "Running on $(hostname), Python: $(python --version)"
PROJ_DIR=$(pwd)

MODEL_PATH="$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean"

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
    --alpha_itc 0 \
    --bf16
    #--lr 2e-5 \
    #--alpha_itc 0.1 \
}

run_one "../src/annotations/jsonl/V0/annotations_V0.jsonl" \
        "train_VL/sft-adapters/qwen2.5-3b-lora-V0" \
        10

run_one "../src/annotations/jsonl/V1-expl/annotations_V1.jsonl" \
        "train_VL/sft-adapters/qwen2.5-3b-lora-V1" \
        10

run_one "../src/annotations/jsonl/V2-aux/annotations_V2.jsonl" \
        "train_VL/sft-adapters/qwen2.5-3b-lora-V2" \
        15

run_one "../src/annotations/jsonl/V3-pic/annotations_V3.jsonl" \
        "train_VL/sft-adapters/qwen2.5-3b-lora-V3" \
        20

echo "All jobs finished."

