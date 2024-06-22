import argparse
import os.path as osp
import yaml
import random
from easydict import EasyDict as edict
import numpy.random as npr
import torch
from utils import edict_2_dict, check_and_create_dir, update
import wandb
import warnings
import os
import numpy as np 
warnings.filterwarnings("ignore")
from glob import glob
import re

def parse_ocr_line(line):
    match = re.match(r'(\[.*?\])\s\(\[(.*?)\],\s\[(.*?)\]\)', line)
    if match:
        indices = match.group(1)
        ocr_word = match.group(2)
        ocr_score = float(match.group(3))
        return indices, ocr_word, ocr_score
    return None, None, None

def parse_clip_line(line):
    match = re.match(r'(\[.*?\])\s(.*)', line)
    if match:
        indices = match.group(1)
        clip_score = float(match.group(2))
        return indices, clip_score
    return None, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config/base.yaml")
    parser.add_argument(
        "--experiment", type=str, default="conformal_0.5_dist_pixel_100_kernel201"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", metavar="DIR", default="output")
    parser.add_argument("--font", type=str, default="none", help="font name")
    parser.add_argument(
        "--semantic_concept", type=str, help="the semantic concept to insert"
    )
    parser.add_argument("--word", type=str, default="none", help="the text to work on")
    parser.add_argument("--script", type=str, default="arabic", help="script")
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="minimal flat 2d vector. lineal color. trending on artstation",
    )
    parser.add_argument(
        "--min_control_points",
        type=int,
        default=0,
        help="min number of control points per path (careful sometimes a dot is a path)",
    )
    parser.add_argument(
        "--operation_mode",
        type=int,
        default=1,
        help="operation mode either ,0 full word , 1 region input ",
    )
    parser.add_argument(
        "--optimized_region",
        type=str,
        default="[0,0]",
        help="the min index and max  inclusive of the region in the word to optimize",
    )
    parser.add_argument(
        "--use_perceptual_loss",
        type=int,
        default=0
    )
    parser.add_argument(
        "--use_ocr_loss",
        type=int,
        default=0
    )


    parser.add_argument(
        "--content_loss_weight", type=float, default=0.001, help="content loss weight"
    )
    parser.add_argument(
        "--style_loss_weight", type=float, default=0.000, help="style loss weight"
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="none")

    parser.add_argument("--use_nst_loss", type=int, default=0)
    parser.add_argument("--use_variational_loss", type=int, default=0)
    parser.add_argument("--variational_loss_weight", type=int, default=1)
    parser.add_argument("--use_blurrer_in_nst", type=int, default=0)
    parser.add_argument("--perceptual_loss_weight", type=float, default=1)
    parser.add_argument("--ocr_loss_weight", type=float, default=1)

    parser.add_argument("--ranking_score", type=int, default=0)
    parser.add_argument("--ranking_num_iter", type=int, default=1)
    parser.add_argument("--ranking", type=int, default=0)


    cfg = edict()
    args = parser.parse_args()
    with open("TOKEN", "r", encoding='utf-8') as f:
        setattr(args, "token", f.read().replace("\n", ""))
    
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.font = args.font
    cfg.semantic_concept = args.semantic_concept
    cfg.word = cfg.semantic_concept if args.word == "none" else args.word
    cfg.letter = cfg.word
    cfg.script = args.script
    cfg.use_nst_loss = args.use_nst_loss
    cfg.use_perceptual_loss =  args.use_perceptual_loss
    cfg.use_ocr_loss =  args.use_ocr_loss

    cfg.ranking_score = args.ranking_score
    cfg.ranking_num_iter = args.ranking_num_iter
    cfg.ranking = args.ranking



    cfg.style_loss_weight = args.style_loss_weight
    cfg.use_variational_loss = args.use_variational_loss
    cfg.variational_loss_weight = args.variational_loss_weight
    cfg.use_blurrer_in_nst = args.use_blurrer_in_nst
    cfg.operation_mode = args.operation_mode

    script_path = f"code/data/fonts/{cfg.script}"
    if cfg.font == "none":
        cfg.font = osp.basename(glob(f"{script_path}/*.ttf")[0])[:-4]

    cfg.caption = f"a {args.semantic_concept}. {args.prompt_suffix}"

    cfg.log_dir = f"{args.log_dir}/{cfg.script}"

    optimized_region = args.optimized_region

    if(cfg.ranking):
        ocr_data = {}
        clip_data = {}

        clip_score_file = f"{cfg.log_dir}/ranking/{cfg.font}_{cfg.word}_clip_score.txt"
        with open(clip_score_file, "r") as file:
            for line in file:
                indices, clip_score = parse_clip_line(line)
                if indices:
                    clip_data[indices] = clip_score

        ocr_score_file = f"{cfg.log_dir}/ranking/{cfg.font}_{cfg.word}_ocr_score.txt"
        with open(ocr_score_file, "r") as file:
            for line in file:
                indices, ocr_word, ocr_score = parse_ocr_line(line)
                if indices:
                    ocr_data[indices] = {'ocr_word': ocr_word, 'ocr_score': ocr_score}

        print(f"OCR Data: {ocr_data}")
        print(f"Clip Data: {clip_data}")
        # Combine the scores and store in a list
        combined_scores = []
        for indices in ocr_data.keys():
            if indices in clip_data:
                total_score = ocr_data[indices]['ocr_score'] + clip_data[indices]
                combined_scores.append((indices, ocr_data[indices]['ocr_word'], total_score))

        # Sort the combined scores based on the total score
        sorted_combined_scores = sorted(combined_scores, key=lambda x: x[2], reverse=True)

        # Print the sorted combined scores
        for indices, ocr_word, total_score in sorted_combined_scores:
            print(f"Indices: {indices}, Total Score: {total_score}")

        if sorted_combined_scores:
            indices, ocr_word, total_score = sorted_combined_scores[0]
        optimized_region = indices

    optimized_range = list(map(int, optimized_region.strip("[]").split(",")))
    if len(optimized_range) == 1:
        optimized_range.append(optimized_range[0])
    assert len(optimized_range) == 2
    cfg.optimized_region = list(range(optimized_range[0], optimized_range[1] + 1))
    optimized_region_name = "".join([str(elem) for elem in cfg.optimized_region])

    # cfg.content_loss_weight = 0.001*len(cfg.optimized_region)
    cfg.content_loss_weight = args.content_loss_weight
    cfg.perceptual_loss_weight = args.perceptual_loss_weight
    cfg.ocr_loss_weight = args.ocr_loss_weight

    cfg.batch_size = args.batch_size
    cfg.token = args.token
    cfg.use_wandb = args.use_wandb
    cfg.wandb_user = args.wandb_user
    cfg.experiment_name = f"{cfg.font}_{cfg.word}_{optimized_region_name if cfg.operation_mode == 1 else 'full_word'}"
    cfg.target = f"code/data/init/{cfg.experiment_name}_scaled"
    if " " in cfg.target:
        cfg.target = cfg.target.replace(" ", "_")

    return cfg


def set_config():
    cfg_arg = parse_args()
    with open(cfg_arg.config, "r") as f:
        cfg_full = yaml.load(f, Loader=yaml.FullLoader)

    # recursively traverse parent_config pointers in the config dicts
    cfg_key = cfg_arg.experiment
    cfgs = [cfg_arg]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get("parent_config", "baseline")

    # allowing children configs to override their parents
    cfg = edict()
    for options in reversed(cfgs):
        update(cfg, options)
    del cfgs

    # set experiment dir
    cfg.signature = (
    f"{cfg.experiment_name}"
    f"_seed_{cfg.seed}"
    f"{'_ocr_loss_' + str(cfg.ocr_loss_weight) if cfg.use_ocr_loss else ''}"
    f"{'_perceptual_loss_' + str(cfg.perceptual_loss_weight) if cfg.use_perceptual_loss else ''}"
    f"{'_content_loss_' + str(cfg.content_loss_weight) if cfg.use_nst_loss else ''}"
    f"{'_useblurNST_' + str(cfg.use_blurrer_in_nst) if cfg.use_blurrer_in_nst else ''}"
    f"{'_angels_loss_' + str(cfg.loss.conformal.angeles_w) if cfg.loss.conformal.use_conformal_loss else ''}"

    )

    cfg.experiment_dir = osp.join(cfg.log_dir, cfg.signature)
    configfile = osp.join(cfg.experiment_dir, "config.yaml")
    print("Config:", cfg)

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(osp.join(configfile), "w") as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(
            project="Word as image",
            entity=cfg.wandb_user,
            name=f"{cfg.semantic_concept}_{cfg.seed}_{cfg.signature}",
            id=wandb.util.generate_id(),
        )

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    else:
        assert False

    return cfg
