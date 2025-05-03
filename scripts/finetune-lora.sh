#!/bin/bash
#SBATCH --job-name=depth-v2-train
#SBATCH --output=logs/.depth_v2_train_%j.out
#SBATCH --error=logs/.depth_v2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
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
    --model vitl \
    --dataset-path /scratch/soccernet/ \
    --sport-name basketball \
    --seed 42 \
    --train-batch-size 2 \
    --val-batch-size 12 \
    --epochs 15 \
    --backbone-lr 0 \
    --head-lr 1e-4 \
    --weight-decay 1e-5 \
    --use-wandb \
    --experiment-name "depth_v2_vitl_lora-64_${SLURM_JOB_ID}" \
    --use-lora \
    --lora-rank 64 \
    --lora-alpha 16 \
    --lora-modules layer1_rn layer2_rn layer3_rn layer4_rn \
    --use-cutmix

# Print end time
echo "End time: $(date)"
