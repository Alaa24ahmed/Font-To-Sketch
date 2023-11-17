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

warnings.filterwarnings("ignore")
from glob import glob


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
        "--use_dot_product_loss",
        type=int,
        default=0,
        help="use dot product loss , helps in preserving structure , hurts the meaning",
    )
    parser.add_argument(
        "--dot_product_loss_weight",
        type=float,
        default=0.2,
        help="dot product loss weight",
    )
    parser.add_argument(
        "--use_content_loss",
        type=int,
        default=0,
        help="use content loss , doubles inference time",
    )
    parser.add_argument(
        "--content_loss_weight", type=float, default=1, help="content loss weight"
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="none")

    cfg = edict()
    args = parser.parse_args()
    with open("TOKEN", "r") as f:
        setattr(args, "token", f.read().replace("\n", ""))

    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.font = args.font
    cfg.semantic_concept = args.semantic_concept
    cfg.word = cfg.semantic_concept if args.word == "none" else args.word
    cfg.letter = cfg.word
    cfg.script = args.script
    cfg.use_dot_product_loss = args.use_dot_product_loss
    cfg.dot_product_loss_weight = args.dot_product_loss_weight
    cfg.use_content_loss = args.use_content_loss
    cfg.content_loss_weight = args.content_loss_weight
    cfg.operation_mode = args.operation_mode

    script_path = f"code/data/fonts/{cfg.script}"
    if cfg.font == "none":
        cfg.font = osp.basename(glob(f"{script_path}/*.ttf")[0])[:-4]

    cfg.caption = f"a {args.semantic_concept}. {args.prompt_suffix}"

    cfg.log_dir = f"{args.log_dir}/{cfg.script}"

    # cfg.optimized_region = list(map(int, args.optimized_region.strip("[]").split(",")))
    optimized_range = list(map(int, args.optimized_region.strip("[]").split(",")))
    assert len(optimized_range) == 2
    cfg.optimized_region = list(range(optimized_range[0], optimized_range[1] + 1))
    optimized_region_name = "".join([str(elem) for elem in cfg.optimized_region])

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
    signature = f"{cfg.experiment_name}_dot_loss_{cfg.dot_product_loss_weight if cfg.use_dot_product_loss else 0}_content_loss{cfg.content_loss_weight if cfg.use_content_loss else 0}_angels_loss{cfg.loss.conformal.angeles_w if cfg.loss.conformal.use_conformal_loss else 0 }_seed_{cfg.seed}_levelOfcc_{cfg.level_of_cc}_sigma_{cfg.loss.tone.pixel_dist_sigma}"
    cfg.experiment_dir = osp.join(cfg.log_dir, signature)
    configfile = osp.join(cfg.experiment_dir, "config.yaml")
    print("Config:", cfg)

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(osp.join(configfile), "w") as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(
            project="Font-To-Image",
            entity=cfg.wandb_user,
            name=f"{cfg.semantic_concept}_{cfg.seed}_{cfg.experiment_name}",
            id=wandb.util.generate_id(),
        )

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only = True)
        torch.backends.cudnn.deterministic = True
    else:
        assert False

    return cfg
