#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=3-00:00
#SBATCH -o /mnt/qb/akata/jstrueber72/logs/clswgan_flo_job_%j.out
#SBATCH -e /mnt/qb/akata/jstrueber72/logs/clswgan_flo_job_%j.err

python /mnt/qb/work/akata/jstrueber72/ZSTTT/clswgan.py --group clswgan --manualSeed 806 --cls_weight 0.1 --syn_num 300 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 97 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --outname flowers --log_online --wandb_key 65954b19f28cc0f35372188d50be8f11cdb79321 --project zsttt

