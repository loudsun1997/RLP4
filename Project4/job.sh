#!/usr/bin/env bash
#SBATCH -A cs525
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 24:00:00
#SBATCH --mem 12G
#SBATCH --job-name="P3"

source activate myenv
python main.py --train_dqn --buffer_type "prioritized_buff" --dqn_type "double dueling dqn" --learning_rate 0.0001 --batch_size 64 --optimizer 'adam' --target_update_freq 10000 --decay_rate 70000 --log_dir "logs/$SLURM_JOB_ID"