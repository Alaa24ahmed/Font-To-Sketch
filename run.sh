seed=42
experiment="default"
word="موسيقى"
concept="music"
script="arabic"
optimized_letter="[0,1]"
font="06"

python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$optimized_letter" --word "$word" --seed $seed --font $font --use_wandb 0 --wandb_user "graduation-word-as-image"

# python code/main.py --experiment "default" --script "arabic" --semantic_concept "duck" --optimized_letter "0,1,2" --word "بطة" --seed 0 --font 06 --use_wandb 0 --wandb_user "graduation-word-as-image"
