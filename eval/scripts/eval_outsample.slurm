#! /bin/bash

#SBATCH --job-name=eval_out
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=publicgpu
#SBATCH --constraint=gpuh100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=75G
#SBATCH --time=03:00:00

module purge
module load python/python-3.11.4
echo "Running on $(hostname), Python: $(python --version)"


srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/qwen2.5-vl-3b-lora-vl-ts1-10ep/checkpoint-580" \
  --output-jsonl "../json_outsample/eval-qwen-ts1-outsample.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts2 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/qwen2.5-vl-3b-lora-vl-ts2-15ep/checkpoint-825" \
  --output-jsonl "../json_outsample/eval-qwen-ts2-outsample.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode ts3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/qwen2.5-vl-3b-lora-vl-ts3-20ep/checkpoint-1060" \
  --output-jsonl "../json_outsample/eval-qwen-ts3-outsample.jsonl" \
  --ban-tool-words

