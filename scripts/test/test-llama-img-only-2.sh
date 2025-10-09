#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu
#SBATCH --mem=82G
#SBATCH --mail-user=xianyuan.liu@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00
#SBATCH --output=out_test_llama_img-only_%j.txt

module load Anaconda3/2024.02-1
module load cuDNN/8.6.0.163-CUDA-11.8.0

cd ~/projects/vlm_post-ocr_correction/

# Activate environment
source activate vlm-post-ocr

# Evaluate model using config
python eval.py --config config/llama-3.2-11b-vision-img-only-2.yaml
