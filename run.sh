seed=42
experiment="default"
word="قلعة"
concept="castle"
script="arabic"
optimized_region="[0,0]"
font="06"
wandb_user="graduation-word-as-image"
use_wandb=1

python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "$optimized_region" --word "$word" --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 1
