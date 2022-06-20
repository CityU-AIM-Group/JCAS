#!/bin/bash
#SBATCH -J Polyp
#SBATCH -o log/SparseReg/Noise_SFDA_1.out    
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -w node30
#SBATCH --partition=team1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch020

cd /home/xiaoqiguo2/Class2affinity/tools_ablation/
python ./SparseReg.py