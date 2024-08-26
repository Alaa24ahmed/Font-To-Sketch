#!/usr/bin/env bash
# export PYTHONIOENCODING=utf-8
seed=1
experiment="default"
# font="15"
# font="IndieFlower-Regular"
# font="HobeauxRococeaux-Sherman"
font="12"
# font="Saira-Regular"KaushanScript-Regular

script="arabic"
# font="GreatVibes-Regular"
# font="BLKCHCRY"
word="سلام"
concept="Human hand making peace sign"

# script="english"
# font="DeliusUnicase-Regular"
wandb_user="graduation-word-as-image"
use_wandb=0

# n=${#word}
# for mx in $(seq 0 $((n-1))); do
#     for mn in $(seq 0 $mx); do 
#         optimized_region="[$mn,$mx]"
#         echo $optimized_region
#         python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --ranking_score 1 --use_ocr_loss 0 --use_conformal_loss 1 --conformal_loss_weight 0.5 --ranking_num_iter 150
#     done
# done


# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "0,0" --word $word --seed $seed --font $font --use_wandb 0 --wandb_user $wandb_user --ranking_score 0 --use_ocr_loss 1 --use_conformal_loss 1 --conformal_loss_weight 0.5 --ranking_num_iter 150
python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "1,1" --word $word --seed $seed --font $font --use_wandb 0 --wandb_user $wandb_user --ranking_score 1 --use_ocr_loss 0 --use_conformal_loss 1 --conformal_loss_weight 0.5 --ranking_num_iter 150
# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "2,2" --word $word --seed $seed --font $font --use_wandb 0 --wandb_user $wandb_user --ranking_score 0 --use_ocr_loss 1 --use_conformal_loss 1 --conformal_loss_weight 0.5 --ranking_num_iter 150

# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "0,2" --word $word --seed $seed --font $font --use_wandb 0 --wandb_user $wandb_user --ranking 1 --use_ocr_loss 1 --use_conformal_loss 1 --conformal_loss_weight 0.5