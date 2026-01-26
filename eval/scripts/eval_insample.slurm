#! /bin/bash

#SBATCH --job-name=eval_insample
#SBATCH --output=qwen_%j.out
#SBATCH --error=qwen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=publicgpu
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --time=07:00:00

module purge
module load python/python-3.11.4 

echo "Running on $(hostname), Python: $(python --version)"

srun --exclusive -N1 -n1 python eval_insample.py \
  --base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --adapter_dir "../../train_VL/qwen2.5-vl-3b-lora-vl-ts1-10ep/checkpoint-580" \
  --data_json		"../../src/annotations/jsonl/annotations_ts1_filtered.jsonl" \
  --val_ratio 0.10 \
  --seed 42 \
  --out_json ../json_insample/eval-qwen-ts1-insample.jsonl

srun --exclusive -N1 -n1 python eval_insample.py \
  --base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --adapter_dir "../../train_VL/qwen2.5-vl-3b-lora-vl-ts2-15ep/checkpoint-825" \
  --data_json		"../../src/annotations/jsonl/annotations_ts2_filtered.jsonl" \
  --val_ratio 0.10 \
  --seed 42 \
  --out_json ../json_insample/eval-qwen-ts2-insample.jsonl

srun --exclusive -N1 -n1 python eval_insample.py \
  --base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --adapter_dir "../../train_VL/qwen2.5-vl-3b-lora-vl-ts3-20ep/checkpoint-1060" \
  --data_json		"../../src/annotations/jsonl/annotations_ts3_filtered.jsonl" \
  --val_ratio 0.10 \
  --seed 42 \
  --out_json ../json_insample/eval-qwen-ts3-insample.jsonl




















#srun --exclusive -N1 -n1 python eval_insample.py \
  #--base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  #--adapter_dir "$HOME/repo-2026/train_VL/qwen2.5-vl-3b-lora-vl-ts2-plain/checkpoint-1100" \
  #--data_json		"$HOME/repo-2026/src/annotations/clean/annot_ts2_plain.jsonl" \
  #--val_ratio 0.10 \
  #--seed 42 \
  #--out_json ../json_insample/eval_vl_insample_ts2-plain-20.jsonl
#



#srun --exclusive -N1 -n1 python eval_insample.py \
  #--base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  #--adapter_dir "$HOME/repo-2026/train_VL/qwen2.5-vl-3b-lora-vl-ts2/checkpoint-770" \
  #--data_json		"$HOME/repo-2026/src/annotations/ready2/annot_ts2.jsonl" \
  #--val_ratio 0.10 \
  #--seed 42 \
  #--out_json results/eval_vl_insample_ts2-77.jsonl
