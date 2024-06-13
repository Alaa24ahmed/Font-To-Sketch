seed=42
experiment="default"
script="arabic"
font="06"
words=("قطة" "ثعلب" "قلعة")
concepts=("cat" "fox" "castle" )
content_loss_weights=(0 0.001 0.002 0.003 0.004 0.005)
perceptual_loss_weights=(0 0.5 1 1.5 2 2.5)
ocr_loss_weights=(0 0.5 1 1.5 2 2.5)
wandb_user="graduation-word-as-image"
use_wandb=1

for i in "${!words[@]}"; do
    n=${#words[i]}
    for mx in $(seq 0 $((n-1))); do
        for mn in $(seq 0 $mx); do 
            optimized_region="[$mn,$mx]"
            for j in "${!content_loss_weights[@]}"; do
                for k in "${!perceptual_loss_weights[@]}"; do
                    for y in "${!ocr_loss_weights[@]}"; do
                        python code/main.py --experiment $experiment --script $script --semantic_concept "${concepts[i]}" --optimized_region "$optimized_region" --word "${words[i]}" --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --content_loss_weight "${content_loss_weights[j]}" --perceptual_loss_weight "${perceptual_loss_weights[k]}" --ocr_loss_weight "${ocr_loss_weights[y]}"
                    done
                done
            done  
        done
    done
done