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

from PIL import Image
import matplotlib.pyplot as plt

def process_image_to_pytorch(batch_size, image):
    image = image.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
    image = image.repeat(batch_size, 1, 1, 1)
    return image

def init_shapes(svg_path):
    svg = f"{svg_path}"
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        svg
    )

    parameters = edict()

    # # path points
    # if trainable.point:
    #     parameters.point = []
    #     for path in shapes_init:
    #         path.points.requires_grad = True
    #         parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters



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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize CLIP scoring
    cfg = type('Config', (), {'script': 'arabic'})()  # Simple config object with 'script' attribute
    clip_score = CLipScoring(cfg, device)


    results = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        video_svg_path = os.path.join(folder_path, "video-svg")
        init_svg_path = os.path.join(folder_path, "svg-init", "init.svg")
        
        # Get the caption from the folder name
        concept = folder.split('_')[1]
        print(concept)
        caption = f"a {concept}. minimal flat 2d vector. lineal color. trending on artstation"

        h, w = 600, 600
        render = pydiffvg.RenderFunction.apply

        # initialize shape
        print("initializing shape")
        shapes, shape_groups, parameters = init_shapes(
            svg_path=init_svg_path
        )

        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img_init = render(w, h, 2, 2, 0, None, *scene_args)
        img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + torch.ones(
            img_init.shape[0], img_init.shape[1], 3, device=device
        ) * (1 - img_init[:, :, 3:4])
        img_init = img_init[:, :, :3]
        # Initialize OcrLoss with init image
        ocr_loss = OcrLoss(img_init)

        # clip_score_res = clip_score.get_loss(tensor_image, caption)
        # # Process init.svg
        # init_clip_score, init_ocr_loss = process_svg(init_svg_path, clip_score, ocr_loss, caption)

        folder_results = []
        for i in range(0, 500, 50):
            svg_file = f"iter{i:04d}.svg"
            svg_path = os.path.join(video_svg_path, svg_file)
            print(svg_path)
            if os.path.exists(svg_path):
                shapes, shape_groups, parameters = init_shapes(
                        svg_path=svg_path
                    )

                scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
                img = render(w, h, 2, 2, 0, None, *scene_args)
                img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
                    img.shape[0], img.shape[1], 3, device=device
                ) * (1 - img[:, :, 3:4])
                img = img[:, :, :3]
                img = process_image_to_pytorch(1, img)

                ocr_loss_value = ocr_loss(img).item()

                

                clip_score_res = clip_score.get_loss(img, caption)
                
                print(clip_score_res, ocr_loss_value)
                # clip_score_res, ocr_loss_value = process_svg(svg_path, clip_score, ocr_loss, caption)
                folder_results.append({
                    'iteration': i,
                    'clip_score': clip_score_res,
                    'ocr_loss': ocr_loss_value
                })

        # Calculate averages
        avg_clip_score = sum(r['clip_score'] for r in folder_results) / len(folder_results)
        avg_ocr_loss = sum(r['ocr_loss'] for r in folder_results) / len(folder_results)

        results.append({
            'folder': folder,
            'avg_clip_score': avg_clip_score,
            'avg_ocr_loss': avg_ocr_loss,
            'iterations': folder_results
        })

    # Write results to CSV
    with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['folder', 'init_clip_score', 'init_ocr_loss', 
                      'avg_clip_score', 'avg_ocr_loss', 'iteration', 'clip_score', 
                      'ocr_loss']
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

if __name__ == "__main__":
    main()