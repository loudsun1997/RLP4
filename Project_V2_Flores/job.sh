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
python main.py --train_dqn --buffer_type "std_buff" --dqn_type "split dqn" --learning_rate 0.00001 --batch_size 128 --optimizer 'rmsprop' --target_update_freq 5000 --decay_rate 80000 --log_dir "logs/$SLURM_JOB_ID"