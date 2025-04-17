#!/bin/bash
#SBATCH --job-name=depth-v2-train
#SBATCH --output=logs/.depth_v2_train_%j.out
#SBATCH --error=logs/.depth_v2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

uv sync

# Activate conda environment (adjust to your environment name)
source .venv/bin/activate
# OR: conda activate depth_env

# Run the training script
python train.py \
    --model vits \
    --dataset_path /scratch/soccernet/depth-basketball/ \
    --sport_name basketball \
    --seed 42 \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --epochs 10 \
    --backbone_lr 1e-6 \
    --head_lr 1e-5 \
    --use_wandb \
    --experiment_name "depth_v2_${SLURM_JOB_ID}"

# Print end time
echo "End time: $(date)"
