#!/bin/bash
#SBATCH -J Polyp
#SBATCH -o log/Baseline.out    
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -w node5

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch020

cd /home/xiaoqiguo2/Class2affinity/tools_ablation/
python ./Baseline.py