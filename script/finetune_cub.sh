#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/akata/jstrueber72/logs/finetune_cub_job_%j.out
#SBATCH -e /mnt/qb/akata/jstrueber72/logs/finetune_cub_job_%j.err

python /mnt/qb/work/akata/jstrueber72/ZSTTT/finetune_backbone.py --log_online --group finetune_resnet50_50_50_split --outname finetune --project zsttt --epochs 90 --learning_rate 1e-4 --save_model --save_valid