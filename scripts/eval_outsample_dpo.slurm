#! /bin/bash

#SBATCH --job-name=100li_out
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=publicgpu
#SBATCH --constraint=gpuh200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH --time=05:00:00

module purge
module load python/python-3.11.4
echo "Running on $(hostname), Python: $(python --version)"

# DPO+ Bradley-Terry:
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpop-bt/checkpoints/epoch_006/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpop-bt-ep6-1.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpop-bt/checkpoints/epoch_006/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpop-bt-ep6-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpop-bt/checkpoints/epoch_006/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpop-bt-ep6-3.jsonl" \
  --ban-tool-words

# DPO+ Plackett-Luce: 
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpop-pl/checkpoints/epoch_008/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpop-pl-ep8-1.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpop-pl/checkpoints/epoch_008/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpop-pl-ep8-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpop-pl/checkpoints/epoch_008/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpop-pl-ep8-3.jsonl" \
  --ban-tool-words

# DPO+- Plackett-Luce: 
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpopm-pl/checkpoints/epoch_009/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpopm-pl-ep9-1.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpopm-pl/checkpoints/epoch_009/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpopm-pl-ep9-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../DPO/dpo-adapters/qwen2.5-vl-3b-v3-dpopm-pl/checkpoints/epoch_009/policy/" \
  --output-jsonl "../json_outsample/DPOP/eval-qwen-v3-outsample-dpopm-pl-ep9-3.jsonl" \
  --ban-tool-words

