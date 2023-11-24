# seed=42
# experiment="default"
# word="موسيقى"
# concept="music"
# script="arabic"
# optimized_region="[0,1]"
# font="06"


seed=42
experiment="default"
word="PANTS"
concept="PANTS"
script="english"
optimized_region="[4,4]"
# font="KaushanScript-Regular"
font="HobeauxRococeaux-Sherman"
# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$optimized_letter" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image"




# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$optimized_letter" --word "$word" --seed $seed --font $font --use_wandb 0 --wandb_user "graduation-word-as-image"
python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "$optimized_region" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image" --use_dot_product_loss 0
python code/main.py --experiment "default" --script "arabic" --semantic_concept "giraffe" --optimized_region "1,1" --word "زرافة" --seed 0 --font 06 --use_wandb 0 --wandb_user "graduation-word-as-image"

python code/main.py --experiment "default" --script "arabic" --semantic_concept "fox" --optimized_region "1,1" --word "ا" --seed 0 --font 06 --use_wandb 0 --wandb_user "graduation-word-as-image"


python code/main.py --experiment "default" --script "english" --semantic_concept "SURFING" --optimized_region "6,6" --word "SURFING" --seed 0 --font "Noteworthy-Bold" --use_wandb 0 --wandb_user "graduation-word-as-image"

KaushanScript-Regular
python code/main.py --experiment "default" --script "english" --semantic_concept "SURFING" --optimized_region "6,6" --word "SURFING" --seed 0 --font "KaushanScript-Regular" --use_wandb 0 --wandb_user "graduation-word-as-image"
HobeauxRococeaux-Sherman

python code/main.py --experiment "default" --script "english" --semantic_concept "SURFING" --optimized_region "0,0" --word "SURFING" --seed 0 --font "HobeauxRococeaux-Sherman" --use_wandb 0 --wandb_user "graduation-word-as-image"


python code/main.py --experiment "default" --script "english" --semantic_concept "PANTS" --optimized_region "0,0" --word "PANTS" --seed 0 --font "HobeauxRococeaux-Sherman" --use_wandb 1 --wandb_user "graduation-word-as-image"

