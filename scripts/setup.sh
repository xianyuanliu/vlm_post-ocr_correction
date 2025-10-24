#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu
#SBATCH --mem=82G
#SBATCH --mail-user=xianyuan.liu@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00
#SBATCH --output=out_setup_%j.txt

module load Anaconda3/2024.02-1
module load cuDNN/8.6.0.163-CUDA-11.8.0

cd /mnt/parscratch/users/ac1xxliu/private/projects/vlm_post-ocr_correction/

# Create environment
conda create -n vlm-post-ocr python=3.10
source activate vlm-post-ocr

# Install requirements
conda install pyarrow
pip install -r requirements.txt

# Create dataset of image patches
python data.py
