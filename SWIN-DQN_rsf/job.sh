#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100
#SBATCH -t 71:59:00
#SBATCH --mem 40G
#SBATCH -p long
#SBATCH --job-name="rezProj4"
#SBATCH --exclude=gpu-5-12

source activate envRL
python3.8 run.py
