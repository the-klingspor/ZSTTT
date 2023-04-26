#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/akata/jstrueber72/logs/finetune_cub_job_%j.out
#SBATCH -e /mnt/qb/akata/jstrueber72/logs/finetune_cub_job_%j.err

python /mnt/qb/work/akata/jstrueber72/ZSTTT/extract_features.py --log_online --group finetune_resnet50 --outname finetune --project zsttt --epochs 60 --learning_rate 3e-4