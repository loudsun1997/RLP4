#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -C A100|V100
#SBATCH -t 24:00:00
#SBATCH --mem 20g
#SBATCH --job-name="P4"

# module load miniconda3/24.1.2/lqdppgt
module load cuda

eval "$(conda shell.bash hook)"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate project33
export PATH=/home/zyang12/miniconda3/envs/project33/bin:$PATH
which python

# python mains.py --train_dqn

# python main.py --train_dqn
# python main.py --test_dqn --record_video
python main.py --config_name 'default_double_dqn_4'