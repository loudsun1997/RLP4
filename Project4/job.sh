#!/bin/bash

#SBATCH -A cs525
#SBATCH -p academic         
#SBATCH -N 1               
#SBATCH -c 8              
#SBATCH --gres=gpu:1        
#SBATCH -t 45:00:00        
#SBATCH --mem=10G           
#SBATCH --job-name="hola_bonita"

source activate myenv

python main.py --train_dqn
# python main.py --test_dqn #--record_video
