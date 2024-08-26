import os
import csv
import torch
from PIL import Image
# from clip_score import CLipScoring
# from ocr_score import OcrLoss
# import cairosvg
import io
import numpy as np

from typing import Mapping
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss, PerceptualLoss
from config import set_config
from ttf import combine_word_mod
from clip_score import CLipScoring
from ocr_score import OcrScoring
from utils import (
    get_data_augs,
    check_and_create_dir,
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word,
    create_video,
)
import wandb
import warnings
import torch.nn as nn
import torchvision.models as models
from losses import NSTLoss, VariationLoss , OcrLoss
import numpy as np

import os
import csv
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from clip_score import CLipScoring
from losses import OcrLoss


def process_image_to_pytorch(image, device):
    image = torch.tensor(np.array(image)).float().to(device) / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    return image

def main():
    base_path = "/home/ahmed.sharshar/Desktop/Alaa/Font-To-Sketch/output_multi/arabic"
    folders = [
        "15_موز_12_seed_42_ocr_loss_0.5_angels_loss_0.5",
        "15_فراولة_01_seed_42_ocr_loss_1.0_angels_loss_0.5",
        "15_طباخ_0_seed_42_ocr_loss_0.5_angels_loss_0.5",
        "15_إيطاليا_234_seed_42_ocr_loss_1.5_angels_loss_0.5",
        "13_قرش_01_seed_42_ocr_loss_1.0_angels_loss_0.5",
        "08_يوغا_23_seed_42_ocr_loss_1.0_angels_loss_0.5",
        "08_طيار_012_seed_42_ocr_loss_1.5_angels_loss_0.5",
        "06_معرفة_1234_seed_42_ocr_loss_2.0_angels_loss_0.5",
        "06_جري_1_seed_42_ocr_loss_0.5_angels_loss_0.5",
        "06_عطف_0_seed_42_ocr_loss_0.5_angels_loss_0.5",
        "06_ثعلب_012_seed_1_ocr_loss_1.5_angels_loss_0.5",
        "05_فلسطين_3_seed_42_ocr_loss_0.5_angels_loss_0.5"
    ]
    folders_concept = {
        "15_موز_12_seed_42_ocr_loss_0.5_angels_loss_0.5" : "bananas",
        "15_فراولة_01_seed_42_ocr_loss_1.0_angels_loss_0.5": "strawberry",
        "15_طباخ_0_seed_42_ocr_loss_0.5_angels_loss_0.5": "Cook",
        "15_إيطاليا_234_seed_42_ocr_loss_1.5_angels_loss_0.5": "Italy",
        "13_قرش_01_seed_42_ocr_loss_1.0_angels_loss_0.5": "Shark",
        "08_يوغا_23_seed_42_ocr_loss_1.0_angels_loss_0.5": "Yoga",
        "08_طيار_012_seed_42_ocr_loss_1.5_angels_loss_0.5": "Pilot",
        "06_معرفة_1234_seed_42_ocr_loss_2.0_angels_loss_0.5": "knowledge",
        "06_جري_1_seed_42_ocr_loss_0.5_angels_loss_0.5": "Running",
        "06_عطف_0_seed_42_ocr_loss_0.5_angels_loss_0.5":"Kindness",
        "06_ثعلب_012_seed_1_ocr_loss_1.5_angels_loss_0.5": "Fox",
        "05_فلسطين_3_seed_42_ocr_loss_0.5_angels_loss_0.5":"Palestine"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize CLIP scoring
    cfg = type('Config', (), {'script': 'arabic'})()  # Simple config object with 'script' attribute
    clip_score = CLipScoring(cfg, device)

    results = []
    iteration_results = {i: {'clip_scores': [], 'ocr_losses': []} for i in [199]}

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        video_png_path = os.path.join(folder_path, "video-png")
        init_png_path = os.path.join(folder_path, "init-png", "init.png")
        
        # Get the caption from the folder name
        concept = folders_concept[folder]
        print(concept)
        caption = f"a {concept}. minimal flat 2d vector. lineal color. trending on artstation"

        # Load init image
        img_init = Image.open(init_png_path).convert('RGB')
        img_init_tensor = process_image_to_pytorch(img_init, device)

        # Initialize OcrLoss with init image
        # ocr_loss = OcrLoss(img_init_tensor.squeeze(0).permute(1, 2, 0))  # Convert to (H, W, C) for OcrLoss
        ocr_loss = OcrLoss(img_init_tensor.squeeze(0).permute(1, 2, 0).to(device))

        folder_results = []
        for i in [199]:
            png_file = f"iter_{i}.png"
            png_path = os.path.join("//home//ahmed.sharshar//Desktop//Alaa//Font-To-Sketch//output_clip_draw", caption, png_file)
            print(png_path)
            if os.path.exists(png_path):
                img = Image.open(png_path).convert('RGB')
                # img_tensor = process_image_to_pytorch(img)
                img_tensor = process_image_to_pytorch(img, device)

                ocr_loss_value = ocr_loss(img_tensor).item()
                clip_score_res = clip_score.get_loss(img_tensor, caption)
                
                print(clip_score_res, ocr_loss_value)
                folder_results.append({
                    'iteration': i,
                    'clip_score': clip_score_res,
                    'ocr_loss': ocr_loss_value
                })
                iteration_results[i]['clip_scores'].append(clip_score_res)
                iteration_results[i]['ocr_losses'].append(ocr_loss_value)

        # Calculate averages
        avg_clip_score = sum(r['clip_score'] for r in folder_results) / len(folder_results)
        avg_ocr_loss = sum(r['ocr_loss'] for r in folder_results) / len(folder_results)

        results.append({
            'folder': folder,
            'avg_clip_score': avg_clip_score,
            'avg_ocr_loss': avg_ocr_loss,
            'iterations': folder_results
        })

    iteration_averages = {}
    for i in [199]:
        if iteration_results[i]['clip_scores'] and iteration_results[i]['ocr_losses']:
            avg_clip = sum(iteration_results[i]['clip_scores']) / len(iteration_results[i]['clip_scores'])
            avg_ocr = sum(iteration_results[i]['ocr_losses']) / len(iteration_results[i]['ocr_losses'])
            iteration_averages[i] = {'avg_clip': avg_clip, 'avg_ocr': avg_ocr}

    # Write results to CSV
    with open('results_clip_draw.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['folder', 'avg_clip_score', 'avg_ocr_loss', 'iteration', 'clip_score', 'ocr_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for iter_result in result['iterations']:
                writer.writerow({
                    'folder': result['folder'],
                    'avg_clip_score': result['avg_clip_score'],
                    'avg_ocr_loss': result['avg_ocr_loss'],
                    'iteration': iter_result['iteration'],
                    'clip_score': iter_result['clip_score'],
                    'ocr_loss': iter_result['ocr_loss']
                })

        # Write iteration averages
        writer.writerow({
            'folder': 'ITERATION AVERAGES',
            'avg_clip_score': '',
            'avg_ocr_loss': '',
            'iteration': '',
            'clip_score': '',
            'ocr_loss': ''
        })
        for iteration, averages in iteration_averages.items():
            writer.writerow({
                'folder': '',
                'avg_clip_score': averages['avg_clip'],
                'avg_ocr_loss': averages['avg_ocr'],
                'iteration': iteration,
                'clip_score': '',
                'ocr_loss': ''
            })

    # Print iteration averages
    print("Iteration Averages:")
    for iteration, averages in iteration_averages.items():
        print(f"Iteration {iteration}:")
        print(f"  Average CLIP score: {averages['avg_clip']}")
        print(f"  Average OCR loss: {averages['avg_ocr']}")

if __name__ == "__main__":
    main()