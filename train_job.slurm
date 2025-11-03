#!/bin/bash
#SBATCH --job-name=plant_health_train
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load 2023
module load Anaconda3/2023.07-2
source ~/.bashrc
source activate plant-health-ai

cd ~/plant-health-ai

python src/train.py

 