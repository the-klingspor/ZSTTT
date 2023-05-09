#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/akata/jstrueber72/logs/clswgan_cub_rot_job_%j.out
#SBATCH -e /mnt/qb/akata/jstrueber72/logs/clswgan_cub_rot_job_%j.err

python /mnt/qb/work/akata/jstrueber72/ZSTTT/clswgan.py --group clswgan_rot --manualSeed 3483 --val_every 1 --cls_weight 0.01 --extract_features --preprocessing --cuda --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --log_online --wandb_key 65954b19f28cc0f35372188d50be8f11cdb79321 --project zsttt
