#!/bin/bash
#SBATCH --job-name=test_efp
#SBATCH --output=logs/test_efp_%j.out
#SBATCH --error=logs/test_efp_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --constraint=xgpe

set -e
set -x

cd ..
mkdir -p logs

source ~/.bashrc
source ~/fyp/fyp_env/bin/activate

python test_efp_equivalence.py --verbose
