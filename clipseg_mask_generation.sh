#!/bin/bash
#SBATCH --job-name=clipseg_waterbird
#SBATCH --output=/home/mila/j/jaewoo.lee/logs/clipseg_waterbird%j.out
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32Gb
#SBATCH --partition=long
#SBATCH --mail-user=jaewoo.lee@mila.quebec
#SBATCH --mail-type=ALL

module load miniconda/3
conda activate dinosam

cd /home/mila/j/jaewoo.lee/projects/text_prompt_sam/clipseg

python clipseg_test.py
