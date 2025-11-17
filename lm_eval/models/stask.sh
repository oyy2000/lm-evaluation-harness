#!/bin/bash 
#SBATCH -J gridsearch_wanda # 作业名 
#SBATCH -p a5000ada
#SBATCH -w c49        # 使用的分区（队列）
#SBATCH --exclusive       # 整个节点只给你一个作业用
#SBATCH --cpus-per-task=16 # CPU 核数（给 Python 线程用） 
#SBATCH --time=2-00:00:00 # 最长运行时间（2 天） 
#SBATCH --output=/home/%u/logs/%x-%j.out
#SBATCH --error=/home/%u/logs/%x-%j.err
source activate fact 
nvidia-smi
python /home/youyang7/projects/lm-evaluation-harness/lm_eval/models/exp.py

