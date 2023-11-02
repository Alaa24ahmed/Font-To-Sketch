seed=42
experiment="default"
word="قطة"
concept="cat"
script="arabic"
USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="graduation-word-as-image"
font="16"

python code/main.py --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --font "$font" --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$word" --word "$word" --seed $seed