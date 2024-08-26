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

warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


def init_shapes(svg_path, trainable: Mapping[str, bool]):
    svg = f"{svg_path}.svg"
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(
        svg
    )

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters


def choose_region(word) -> (int, int):
    pass


def process_image_to_pytorch(batch_size, image):
    image = image.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
    image = image.repeat(batch_size, 1, 1, 1)
    return image

def get_scheduler(step=None):
    if step is not None:
        return np.exp(-(1 / 5) * ((step - 300) / (20)) ** 2)
    else:
        return 1


def linear_scheduler(step, num_iter=500, change_point=300, rate1=2.0, base=0.0):
    rate2 = rate1 * 2
    if step <= change_point:
        # Linearly increase from base to rate1
        return ((rate1 - base) / change_point) * step + base
    else:
        # Linearly increase from rate1 to rate2 to the end
        return ((rate2 - rate1) / (num_iter - change_point)) * (step - change_point) + rate1

if __name__ == "__main__":
    cfg = set_config()
    # print(cfg)
    print("font: ", cfg.font)
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()
    print(f"device {device}")
    print("preprocessing")
    
    
    preprocess(
        cfg.font,
        cfg.word,
        cfg.optimized_region,
        cfg.experiment_name,
        cfg.script,
        cfg.level_of_cc,
    )
    h, w = cfg.render_size, cfg.render_size
    data_augs = get_data_augs(cfg.cut_size)
    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print("initializing shape")
    shapes, shape_groups, parameters = init_shapes(
        svg_path=cfg.target, trainable=cfg.trainable
    )

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + torch.ones(
        img_init.shape[0], img_init.shape[1], 3, device=device
    ) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]

    # if cfg.use_ocr_loss:
    ocr_loss =OcrLoss(img_init)

    if cfg.loss.use_sds_loss :
        sds_loss = SDSLoss(cfg, device)
        im_init = process_image_to_pytorch(cfg.batch_size, img_init)
        im_init = data_augs.forward(im_init)
        sds_loss.set_image_init(im_init)

    if cfg.use_nst_loss:
        nst_loss = NSTLoss(
            cfg, process_image_to_pytorch(cfg.batch_size, img_init), device
        )

    if cfg.use_variational_loss:
        variational_loss = VariationLoss(cfg)

    im_init = im_init.squeeze(0).permute(1, 2, 0)

    imshow = im_init.detach().cpu()
    filename = os.path.join(cfg.experiment_dir, "init-png", "init.png")
    pydiffvg.imwrite(imshow, filename, gamma=gamma)
    print("Saved image in directory: ", filename)

    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.use_tone_loss:
        print("initializing tone loss")
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.save.init:
        print("saving init")
        filename = os.path.join(cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    if(cfg.ranking_score):
        num_iter = cfg.ranking_num_iter

    pg = [{"params": parameters["point"], "lr": cfg.lr_base["point"]}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    # if cfg.loss.conformal.use_conformal_loss:
    if cfg.use_conformal_loss:
        print("initializing conformal loss")
        conformal_loss = ConformalLoss(
            cfg, parameters, device, cfg.optimized_region, shape_groups
        )

    if(cfg.use_perceptual_loss):
        perceptual_loss = PerceptualLoss(cfg)
        perceptual_loss.set_image_init(img_init)

    lr_lambda = (
        lambda step: learning_rate_decay(
            step,
            cfg.lr.lr_init,
            cfg.lr.lr_final,
            num_iter,
            lr_delay_steps=cfg.lr.lr_delay_steps,
            lr_delay_mult=cfg.lr.lr_delay_mult,
        )
        / cfg.lr.lr_init
    )

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)
    print(cfg.caption)
    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        if cfg.save.video and (
            step % cfg.save.video_frame_freq == 0 or step == num_iter - 1
        ):
            save_image(
                img,
                os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"),
                gamma,
            )
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg"
            )
            check_and_create_dir(filename)
            save_svg.save_svg(filename, w, h, shapes, shape_groups)

        # img_path = f"{cfg.experiment_dir}/{cfg.font}_{cfg.word}_{cfg.optimized_region}.png"
        # image = Image.open(img_path)
        
    
        x = process_image_to_pytorch(cfg.batch_size, img)
        x_aug = x
        x_aug = data_augs.forward(x) 
       
            
        # compute diffusion loss per pixel
        loss = 0 
        
        sds_loss_res = sds_loss(x_aug)
        loss = sds_loss_res
        print(f"sds loss: {sds_loss_res}")
    
        if cfg.loss.tone.use_tone_loss:
            tone_loss_res = tone_loss(x, step)
            print(f"tone loss: {tone_loss_res}")
            loss = loss + tone_loss_res

        # if cfg.loss.conformal.use_conformal_loss:
        if cfg.use_conformal_loss:
            loss_angles = conformal_loss()
            # loss_angles = cfg.loss.conformal.angeles_w * loss_angles
            loss_angles = cfg.conformal_loss_weight * loss_angles
            print(f"loss_angles: {loss_angles}")
            if(not torch.isnan(loss_angles)):
                print("here")
                print(type(loss_angles))
                loss = loss + loss_angles

        if(cfg.use_perceptual_loss):

            perceptual_loss_res = cfg.perceptual_loss_weight * perceptual_loss(x) 
            print(f"perceptual loss: {perceptual_loss_res}")
            loss = loss + perceptual_loss_res


        if cfg.use_nst_loss:
            loss_content, loss_style = nst_loss(x)
            loss_content = cfg.content_loss_weight * loss_content
            loss += loss_content
            
            print(f"loss_content: {loss_content}")

        if cfg.use_variational_loss:
            loss_variational = variational_loss(x)
            loss = loss + cfg.variational_loss_weight * loss_variational
            print(f"loss_variational: {loss_variational}")

        
        if cfg.use_ocr_loss:
            loss_ocr = ocr_loss(x) * cfg.ocr_loss_weight
            loss = loss + loss_ocr
            print(f"loss_ocr: {loss_ocr}" )
            
        
        
        #print and deal with the confidencies first 
        # ocr_score_res = ocr_score.get_score(x)
        # print(f"ocr score: {ocr_score_res}")

        # total_score = (1 - α) * clip_score_res + α * ocr_score_res

        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]["lr"]}, step=step)
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()
            wandb.log({"sds_loss": sds_loss_res.item()}, step=step)
            if cfg.loss.tone.use_tone_loss:
                wandb.log({"tone_loss": tone_loss_res.item()}, step=step)
            if cfg.loss.conformal.use_conformal_loss:
                wandb.log({"loss_angles": loss_angles.item()}, step=step)
            if cfg.use_nst_loss and cfg.content_loss_weight > 0.0:
                wandb.log({"loss_content": loss_content.item()}, step=step)
            if cfg.use_variational_loss:
                wandb.log({"loss_variational": loss_variational.item()}, step=step)
            if cfg.use_perceptual_loss and cfg.perceptual_loss_weight > 0.0 :
                wandb.log({"perceptual_loss": perceptual_loss_res.item()}, step=step)
            if cfg.use_ocr_loss and cfg.ocr_loss_weight > 0.0:
                wandb.log({"ocr_loss": loss_ocr.item()}, step=step)
            clip_score = CLipScoring(cfg, device)
            clip_score_res = clip_score.get_loss(x, cfg.caption)
            ocr_loss_graph = ocr_loss(x).item()
            # readability_score = 1 - loss_ocr.item()
            wandb.log({"clip_score": clip_score_res, "ocr_loss_graph": ocr_loss_graph}, step=step)
        
        t_range.set_postfix({"loss": loss.item()})
        print(f"loss: {loss}")
        print(f"loss_item: {loss.item()}")
        torch.cuda.empty_cache()
        loss.backward()
        optim.step()
        scheduler.step()
        torch.cuda.empty_cache()

    filename = os.path.join(cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(filename, w, h, shapes, shape_groups)

    combine_word(
        cfg.target,
        cfg.word,
        cfg.optimized_region,
        cfg.font,
        cfg.experiment_dir,
        cfg.script, h, w,
    )

    if cfg.save.image:
        filename = os.path.join(cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)

    if(cfg.ranking_score):
        # ocr_score = OcrScoring(cfg, device)
        # image = Image.open( f"{cfg.experiment_dir}/{cfg.font}_{cfg.word}_{cfg.optimized_region}.png")
        print("hereeee")
        ocr_score_res = ocr_loss(x) * 1

        # ocr_score_res = ocr_score.get_score(image)
        print(f"ocr score: {ocr_score_res}")
        clip_score = CLipScoring(cfg, device)
        clip_score_res = clip_score.get_loss(x, cfg.caption)
        print(f"clip score: {clip_score_res}")
        check_and_create_dir(f"{cfg.log_dir}/ranking/{cfg.font}_{cfg.word}_clip_score.txt")
        with open(f"{cfg.log_dir}/ranking/{cfg.font}_{cfg.word}_clip_score.txt", 'a', encoding="utf-8") as file:
            file.write(str(cfg.optimized_region) + " " + str(clip_score_res) + '\n')
        with open(f"{cfg.log_dir}/ranking/{cfg.font}_{cfg.word}_ocr_score.txt", 'a', encoding="utf-8") as file:
            file.write(str(cfg.optimized_region) + " " + str(ocr_score_res.item()) + '\n')
        


    ocr_score = OcrScoring(cfg, device)
    image = Image.open( f"{cfg.experiment_dir}/{cfg.font}_{cfg.word}_{cfg.optimized_region}.png")
    ocr_score_res = ocr_score.get_score(image)
    print(f"ocr score: {ocr_score_res}")
    clip_score = CLipScoring(cfg, device)
    clip_score_res = clip_score.get_loss(x, cfg.caption)
    print(f"clip score: {clip_score_res}")



    if cfg.use_wandb:    
        img_path = f"{cfg.experiment_dir}/{cfg.font}_{cfg.word}_{cfg.optimized_region}.png"
        pil_im = Image.open(img_path)
        plt = plt.imshow(pil_im)
        wandb.log({"final": wandb.Image(plt)}, step=500)
        plt.close()

    if(not cfg.ranking_score):
        if cfg.save.video:
            print("saving video")
            create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)
        

    if cfg.use_wandb:
        wandb.finish()
