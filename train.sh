#!/bin/bash

#SBATCH --job-name=Evidential_Uncertainty_Selection
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH -w agi1
#SBATCH -p batch
#SBATCH -t 8:00:00
#SBATCH -o %x_%j.out
#SBTACH -e %x_%j.err

source /root/data/init.sh
conda activate uncertainty

data_path="/local_datasets/"
output_path="./output/"

CUDA_VISIBLE_DEVICES=3 python main.py \
    --data_path $data_path \
    --output_path $output_path \
    --model su_vit_tiny_patch16_224 \
    --dataset mnist \
    --base_keep_rate 0.7 \
    --uncertainty_keep_rate 0.8 \
    --epoch 30 \
    --batch_size 256 \
    --input_size 224 \

echo "Output path: $output_path"