#!/bin/bash
 
#SBATCH --job-name=Fetal_femur
#SBATCH --account=st-sdena-1-gpu
#SBATCH --nodes=1                  
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4                         
#SBATCH --mem=32G                  
#SBATCH --time=70:00:00             
#SBATCH --gpus-per-node=2
#SBATCH --output=out-Fetal_femur.txt         
#SBATCH --error=error-Fetal_femur.txt           
#SBATCH --mail-user=miladyz@student.ubc.ca
#SBATCH --mail-type=ALL    
#SBATCH --constraint=gpu_mem_32

module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
source /scratch/st-sdena-1/miladyz/env_py3.8.10/bin/activate  
cd $SLURM_SUBMIT_DIR
pwd
nvidia-smi

export HOME="/scratch/st-sdena-1/miladyz"

python train_unconditional.py --class_ultra "Fetal femur" --num_epochs 400