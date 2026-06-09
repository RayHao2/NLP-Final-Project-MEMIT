#!/bin/bash
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=2-0
#SBATCH --job-name=eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
mkdir -p logs



export HF_HOME="${HF_HOME:-/nfs/guille/eecs_research/soundbendor/sentiment/hf_cache}"


source /nfs/guille/eecs_research/soundbendor/sentiment/miniconda3/bin/activate
source activate memit
nvidia-smi
# quick gates
python eval_gptj.py --dataset counterfact --limit 20  --skip-baseline --conditions \
    atomic.naive \
    atomic.method1.a0p001 \
    atomic.method1.a0p01 \
    atomic.method1.a0p1 \
    atomic.method1.a1 \
    corrective.naive \
    corrective.method1.a0p001 \
    corrective.method1.a0p01 \
    corrective.method1.a0p1 \
    corrective.method1.a1
python eval_gptj.py --dataset counterfact --limit 20 --noisy --skip-baseline --conditions \
    atomic.naive \
    atomic.method1.a0p001 \
    atomic.method1.a0p01 \
    atomic.method1.a0p1 \
    atomic.method1.a1 \
    corrective.naive \
    corrective.method1.a0p001 \
    corrective.method1.a0p01 \
    corrective.method1.a0p1 \
    corrective.method1.a1

# then the real runs
python eval_gptj.py --dataset counterfact --conditions \
    atomic.naive \
    atomic.method1.a0p001 \
    atomic.method1.a0p01 \
    atomic.method1.a0p1 \
    atomic.method1.a1 \
    corrective.naive \
    corrective.method1.a0p001 \
    corrective.method1.a0p01 \
    corrective.method1.a0p1 \
    corrective.method1.a1
python eval_gptj.py --dataset counterfact --noisy --skip-baseline --conditions \
    atomic.naive \
    atomic.method1.a0p001 \
    atomic.method1.a0p01 \
    atomic.method1.a0p1 \
    atomic.method1.a1 \
    corrective.naive \
    corrective.method1.a0p001 \
    corrective.method1.a0p01 \
    corrective.method1.a0p1 \
    corrective.method1.a1