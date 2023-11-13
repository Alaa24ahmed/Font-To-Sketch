#Test a word with all possible optimized regions with and without dot product loss , dont try this at home unless you want a fried GPU

seed=42
experiment="default"
script="arabic"
font="06"
words=("قطة" ,"ثعلب", "قهوة","حوت", "كلب", "زرافة" ,"فهد")
concepts=("cat" ,"fox", "whale","coffee", "dog", "giraffe", "leopard")


for i in "${!words[@]}"; do
    n=${#words[i]}
    for mx in $(seq 0 $n); do 
        for mn in $(seq 0 $mx); do 
            optimized_region="[$mn,$mx]"
            python code/main.py --experiment $experiment --script $script --semantic_concept "${concepts[i]}" --optimized_region "$optimized_region" --word "${word[i]}" --seed $seed --font $font --use_wandb 0 --wandb_user "graduation-word-as-image"
            python code/main.py --experiment $experiment --script $script --semantic_concept "${concepts[i]}" --optimized_region "$optimized_region" --word "${word[i]}" --seed $seed --font $font --use_wandb 0 --wandb_user "graduation-word-as-image" --used_dot_product_loss 1
        done
        
    done

done 

