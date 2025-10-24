#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu
#SBATCH --mem=82G
#SBATCH --mail-user=xianyuan.liu@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00
#SBATCH --output=out_train_llama_img-only_%j.txt

module load Anaconda3/2024.02-1
module load cuDNN/8.6.0.163-CUDA-11.8.0

cd /mnt/parscratch/users/ac1xxliu/private/projects/vlm_post-ocr_correction/

# Activate environment
source activate vlm-post-ocr

# Train/Evaluate model using config
for i in {1..5}
do
    echo ">>> Run $i starting at $(date): config/llama-3.2-11b-vision-img-only-$i.yaml"
    python finetune.py --config config/llama-3.2-11b-vision-img-only-$i.yaml
    python eval.py --config config/llama-3.2-11b-vision-img-only-$i.yaml
    echo ">>> Run $i finished at $(date)"
done
