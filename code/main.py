from typing import Mapping
import os
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import SDSLoss, ToneLoss, ConformalLoss
from config import set_config
from ttf import combine_word_mod
from utils import (
    check_and_create_dir,
    get_data_augs,
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
from losses import ContentLoss

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


if __name__ == "__main__":
    cfg = set_config()
    print(cfg)
    print("font: ", cfg.font)
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()
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

    if cfg.loss.use_sds_loss:
        sds_loss = SDSLoss(cfg, device)
        im_init = process_image_to_pytorch(cfg.batch_size, img_init)
        im_init = data_augs.forward(im_init)
        sds_loss.set_image_init(im_init)

    if cfg.use_content_loss:
        content_loss = ContentLoss(cfg, im_init)

    im_init = im_init.squeeze(0).permute(1, 2, 0)

    imshow = im_init.detach().cpu()
    filename = os.path.join(cfg.experiment_dir, "init-png", "init.png")
    pydiffvg.imwrite(imshow, filename, gamma=gamma)
    print("Saved image in directory: ", filename)

    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        print("initializing tone loss")
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.save.init:
        print("saving init")
        filename = os.path.join(cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{"params": parameters["point"], "lr": cfg.lr_base["point"]}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        print("initializing conformal loss")
        conformal_loss = ConformalLoss(
            cfg, parameters, device, cfg.optimized_region, shape_groups
        )

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

        x = process_image_to_pytorch(cfg.batch_size, img)
        x_aug = data_augs.forward(x)

        # compute diffusion loss per pixel
        sds_loss_res = sds_loss(x_aug)
        loss = sds_loss_res

        if cfg.loss.tone.use_tone_loss:
            tone_loss_res = tone_loss(x, step)
            loss = loss + tone_loss_res

        if cfg.loss.conformal.use_conformal_loss:
            loss_angles = conformal_loss()
            loss_angles = cfg.loss.conformal.angeles_w * loss_angles
            print(f"loss_angles: {loss_angles}")
            loss = loss + loss_angles

        if cfg.use_content_loss:
            loss_content = content_loss(x_aug)
            loss_content = cfg.loss.content_weight * loss_content
            print(f"loss_content: {loss_content}")
            loss = loss + loss_content

        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]["lr"]}, step=step)
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()
            wandb.log({"sds_loss": sds_loss_res.item()}, step=step)
            wandb.log({"dist_loss": tone_loss_res}, step=step)
            wandb.log({"loss_angles": loss_angles}, step=step)
            wandb.log({"loss_content": loss_content}, step=step)

        t_range.set_postfix({"loss": loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()

    filename = os.path.join(cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(filename, w, h, shapes, shape_groups)

    combine_word_mod(
        cfg.target, cfg.word, cfg.optimized_region, cfg.font, cfg.experiment_dir
    )
    if cfg.save.image:
        filename = os.path.join(cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()
