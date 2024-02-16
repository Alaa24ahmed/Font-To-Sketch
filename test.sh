seed=42
experiment="default"
script="arabic"
font="06"
words=("قطة" "ثعلب" "قهوة""حوت" "كلب" "زرافة" "فهد")
concepts=("cat" "fox" "coffee" "whale" "dog" "giraffe" "leopard")
wandb_user="graduation-word-as-image"
use_wandb=1

for i in "${!words[@]}"; do
    n=${#words[i]}
    for mx in $(seq 0 $((n-1))); do
        for mn in $(seq 0 $mx); do 
            optimized_region="[$mn,$mx]"
            python code/main.py --experiment $experiment --script $script --semantic_concept "${concepts[i]}" --optimized_region "$optimized_region" --word "${words[i]}" --seed $seed --font $font --use_wandb $use_wandb --wandb_user $wandb_user --use_nst_loss 1
        done
    done
done 
