# seed=42
# experiment="default"
# word="موسيقى"
# concept="music"
# script="arabic"
# optimized_region="[0,1]"
font="06"


seed=42
experiment="default"
word="ثعلب"
concept="fox"
script="arabic"
optimized_region="[0,0]"
# font="KaushanScript-Regular"
# font="HobeauxRococeaux-Sherman"
# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$optimized_letter" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image"




# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$optimized_letter" --word "$word" --seed $seed --font $font --use_wandb 0 --wandb_user "graduation-word-as-image"
python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "$optimized_region" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image" --use_dot_product_loss 0 --use_content_loss 1 
optimized_region="[1,1]"
python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "$optimized_region" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image" --use_dot_product_loss 0 --use_content_loss 1 
optimized_region="[2,2]"
python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "$optimized_region" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image" --use_dot_product_loss 0 --use_content_loss 1 
optimized_region="[3,3]"
python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region "$optimized_region" --word "$word" --seed $seed --font $font --use_wandb 1 --wandb_user "graduation-word-as-image" --use_dot_product_loss 0 --use_content_loss 1 
