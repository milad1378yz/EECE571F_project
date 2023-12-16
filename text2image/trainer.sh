#!/bin/bash
 
#SBATCH --job-name=train-text-image-1
#SBATCH --account=st-sdena-1-gpu
#SBATCH --nodes=1                  
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4                         
#SBATCH --mem=32G                  
#SBATCH --time=70:00:00             
#SBATCH --gpus-per-node=2
#SBATCH --output=output-text-image-2.txt         
#SBATCH --error=error-text-image-2.txt           
#SBATCH --mail-user=miladyz@student.ubc.ca
#SBATCH --mail-type=ALL    
#SBATCH --constraint=gpu_mem_32

# module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
# source /scratch/st-sdena-1/miladyz/env_py3.8.10/bin/activate  
# cd $SLURM_SUBMIT_DIR
# pwd
# nvidia-smi

export HOME="/scratch/st-sdena-1/miladyz"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="data/FETAL_PLANES_ZENODO"

export OUTPUT_DIR="results/"


accelerate launch  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=30001 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \


