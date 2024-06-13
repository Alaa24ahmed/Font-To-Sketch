seed=42
experiment="default"
word="قلعة"
concept="castle"
script="arabic"
optimized_region="0,3"
font="06"
wandb_user="graduation-word-as-image"
use_wandb=0

python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0


seed=42
experiment="default"
word="قلعة"
concept="castle"
script="arabic"
optimized_region="1,3"
font="06"
wandb_user="graduation-word-as-image"
use_wandb=0

python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0


seed=42
experiment="default"
word="قلعة"
concept="castle"
script="arabic"
optimized_region="2,3"
font="06"
wandb_user="graduation-word-as-image"
use_wandb=0

python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 1
python code/main.py --experiment $experiment --script $script --semantic_concept $concept --optimized_region $optimized_region --word $word --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 0 --use_perceptual_loss 0 --use_ocr_loss 0


