#! /bin/bash

#SBATCH --job-name=eval_out
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

# V0:
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v0 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V0/checkpoint-580" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v0-outsample-3.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v0 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V0/checkpoint-580" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v0-outsample-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v0 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V0/checkpoint-580" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v0-outsample-1.jsonl" \
  --ban-tool-words

# V1:
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v1 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V1/checkpoint-580" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v1-outsample-3.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v1 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V1/checkpoint-580" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v1-outsample-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v1 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V1/checkpoint-580" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v1-outsample-1.jsonl" \
  --ban-tool-words
  
# V2:
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v2 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V2/checkpoint-795" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v2-outsample-3.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v2 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V2/checkpoint-795" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v2-outsample-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v2 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V2/checkpoint-795" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v2-outsample-1.jsonl" \
  --ban-tool-words

# V3:
srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V3/checkpoint-1060" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v3-outsample-3.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V3/checkpoint-1060" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v3-outsample-2.jsonl" \
  --ban-tool-words

srun --exclusive -N1 -n1 python eval_outsample.py \
  --mode v3 \
  --vl-model-path "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --csv-dir "../../all_data/UCR_dataset/csv_data/" \
  --img-dir "../../all_data/UCR_dataset/figs_data/"  \
  --lora-adapter-path "../../train_VL/sft-adapters/qwen2.5-3b-lora-V3/checkpoint-1060" \
  --output-jsonl "../json_outsample/V0-3/eval-qwen-v3-outsample-1.jsonl" \
  --ban-tool-words


