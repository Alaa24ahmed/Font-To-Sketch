seed=42
experiment="default"
# word="CASTLE"
# concept="castle"

# script="english"
# font="HobeauxRococeaux-Sherman"
# word="Avocado"
# concept="Avocado"

script="arabic"
font="15"
word="موز"
concept="banana"

optimized_region="1,2"

# font="IndieFlower-Regular"
# font="LuckiestGuy-Regular"
wandb_user="graduation-word-as-image"
# use_wandb=1

python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb 0 --wandb_user $wandb_user --use_ocr_loss 0 --use_tone_loss 1 --use_conformal_loss 1 --conformal_loss_weight 0.5

# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "0,2" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "0,0" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0

# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "1,1" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "1,1" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0



# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "2,2" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "2,2" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0


# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "3,3" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
# python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region "3,3" --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0


