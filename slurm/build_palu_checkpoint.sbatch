#!/bin/bash
#SBATCH --job-name=build_palu_ckpt
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --qos=spot

set -euo pipefail

# Some user bashrc files assume interactive shell variables exist.
set +u
source ~/.bashrc
source /home/lyut/miniforge3/etc/profile.d/conda.sh
conda activate palu
set -u
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR/third_party/palu:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

python scripts/build_palu_checkpoint.py "$@"
