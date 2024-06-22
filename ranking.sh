#!/usr/bin/env bash
export PYTHONIOENCODING=utf-8
seed=42
experiment="default"
script="arabic"
font="06"
word="قطة"
concept="cat"
wandb_user="graduation-word-as-image"
use_wandb=0

n=${#word}
for mx in $(seq 0 $((n-1))); do
    for mn in $(seq 0 $mx); do 
        optimized_region="[$mn,$mx]"
        echo $optimized_region
        conda run -n word python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --ranking_score 1
    done
done
conda run -n word python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --ranking 1