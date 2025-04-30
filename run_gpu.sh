#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=gpu_job.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load anaconda3/2024.06/gcc-13.2.0
mv $HOME/.conda $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda
source activate sae
module load cuda/12.2.1/gcc-11.2.0
module load gcc/11.2.0/gcc-4.8.5
pip install -r requirements.txt
python trainSAE.py