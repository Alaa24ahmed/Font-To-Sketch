seed=42
experiment="default"
word="موسيقى"
concept="music"
script="arabic"

# python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$word" --word "$word" --seed $seed

# python code/main.py --experiment "default" --script "arabic" --semantic_concept "dog" --optimized_letter "كلب" --word "كلب" --seed 0 --use_wandb 0 --wandb_user "graduation-word-as-image"

python code/main.py --experiment "default" --script "arabic" --semantic_concept "Horse" --optimized_letter 0 --word "حصان" --seed 0 --use_wandb 1 --wandb_user "graduation-word-as-image"
python code/main.py --experiment "default" --script "arabic" --semantic_concept "Horse" --optimized_letter 1 --word "حصان" --seed 0 --use_wandb 1 --wandb_user "graduation-word-as-image"
python code/main.py --experiment "default" --script "arabic" --semantic_concept "Horse" --optimized_letter 2 --word "حصان" --seed 0 --use_wandb 1 --wandb_user "graduation-word-as-image"
python code/main.py --experiment "default" --script "arabic" --semantic_concept "Horse" --optimized_letter 3 --word "حصان" --seed 0 --use_wandb 1 --wandb_user "graduation-word-as-image"