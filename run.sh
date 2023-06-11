seed=42
experiment="default"
word="موسيقى"
concept="music"
script="arabic"

python code/main.py --experiment $experiment --script $script --semantic_concept "$concept" --optimized_letter "$word" --word "$word" --seed $seed