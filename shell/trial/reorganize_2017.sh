#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


set -e
set -x

cd /home/e/e0968951/fyp/Surgical-SAM-2/tools

python convert.py
