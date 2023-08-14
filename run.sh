seed=42
experiment="default"
word="موسيقى"
concept="music"
script="arabic"
font="16"

python code/main.py --experiment $experiment --script $script --font "$font" --semantic_concept "$concept" --optimized_letter "$word" --word "$word" --seed $seed