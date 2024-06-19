import collections.abc
import os
import os.path as osp
from torch import nn
import kornia.augmentation as K
import pydiffvg
import save_svg
import cv2
from ttf import font_string_to_svgs, normalize_svg_size, extract_svg_paths
import torch
import numpy as np


def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append(edict_2_dict(x[i]))
        return xnew
    else:
        return x


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


def update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def preprocess(font, word, letters, experiment_name, script, level_of_cc=1):
    safe_arabic_fonts = {"01", "02", "03", "05", "06", "08", "11", "12", "13", "14", "15", "16", "18", "19", "20"}
    target_cp = None

    if level_of_cc == 0:
        target_cp = None
    else:
        if script == "english":
            target_cp = {
                "A": 120,
                "B": 120,
                "C": 100,
                "D": 100,
                "E": 120,
                "F": 120,
                "G": 120,
                "H": 120,
                "I": 35,
                "J": 80,
                "K": 100,
                "L": 80,
                "M": 100,
                "N": 100,
                "O": 100,
                "P": 120,
                "Q": 120,
                "R": 130,
                "S": 110,
                "T": 90,
                "U": 100,
                "V": 100,
                "W": 100,
                "X": 130,
                "Y": 120,
                "Z": 120,
                "a": 120,
                "b": 120,
                "c": 100,
                "d": 100,
                "e": 120,
                "f": 120,
                "g": 120,
                "h": 120,
                "i": 35,
                "j": 80,
                "k": 100,
                "l": 80,
                "m": 100,
                "n": 100,
                "o": 100,
                "p": 120,
                "q": 120,
                "r": 130,
                "s": 110,
                "t": 90,
                "u": 100,
                "v": 100,
                "w": 100,
                "x": 130,
                "y": 120,
                "z": 120,
            }
        elif script == "arabic" and font in safe_arabic_fonts:
            # Arabic letters (example with placeholders, adjust the counts as needed)
            target_cp = {
                "ا": 120,
                "ب": 80,
                "ت": 80,
                "ث": 80,
                "ج": 80,
                "ح": 80,
                "خ": 80,
                "د": 60,
                "ذ": 60,
                "ر": 60,
                "ز": 60,
                "س": 100,
                "ش": 100,
                "ص": 100,
                "ض": 100,
                "ط": 120,
                "ظ": 120,
                "ع": 100,
                "غ": 100,
                "ف": 100,
                "ق": 100,
                "ك": 100,
                "ل": 100,
                "م": 100,
                "ن": 100,
                "ه": 100,
                "و": 100,
                "ي": 100,
                "ة": 60,
            }
        if(target_cp):
            target_cp = {k: v * level_of_cc for k, v in target_cp.items() } if target_cp else None

    print(f"======= {font} =======")
    font_path = f"code/data/fonts/{script}/{font}.ttf"
    init_path = f"code/data/init"
    subdivision_thresh = None

    print("word: ", word)

    if not os.path.isdir(init_path):
        os.mkdir(init_path)

    svg_path = f"{init_path}/{experiment_name}"
    svg_path = svg_path.replace(" ", "_")

    font_string_to_svgs(
        svg_path,
        font_path,
        word,
        target_control=target_cp,
        subdivision_thresh=subdivision_thresh,
    )
    # normalize_svg_size(svg_path)

    if "full_word" not in svg_path:
        extract_svg_paths(svg_path, letters, script)
    print("Done preprocess")


# Try removing the randomnes here or the augmentation entirely to see if that makes other losses converge
def get_data_augs(cut_size):
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
    augmentations.append(
        K.RandomCrop(
            size=(cut_size, cut_size), pad_if_needed=True, padding_mode="reflect", p=1.0
        )
    )
    return nn.Sequential(*augmentations)


"""pytorch adaptation of https://github.com/google/mipnerf"""


def learning_rate_decay(
    step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1
):
    """Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def save_image(img, filename, gamma=1):
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)


# def get_letter_ids(letter, word, shape_groups):
#     for group, l in zip(shape_groups, word):
#         if l == letter:
#             return group.shape_ids
        
def get_letter_ids(letter, word, shape_groups):
    return [int(x) for x in shape_groups[letter].shape_ids]


def combine_word(svg_path, word, letter, font, experiment_dir, script):
    svg_path = svg_path[:-7]
    word_svg= f"{svg_path}.svg"
    normalize_svg_size(svg_path)
    word_svg_scaled = f"{svg_path}_scaled.svg"
    # word_svg_scaled = f"./code/data/init/{font}_{word}_scaled.svg"
    (
        canvas_width_word,
        canvas_height_word,
        shapes_word,
        shape_groups_word,
    ) = pydiffvg.svg_to_scene(word_svg_scaled)

    letter_ids = []
    for l in letter:
        if script == "english":
            letter_ids += get_letter_ids(l, word, shape_groups_word)
        elif script == "arabic":
            letter_ids += get_letter_ids(len(word)-l-1, word, shape_groups_word)
        else:
            letter_ids += get_letter_ids(l, word, shape_groups_word)
    print(letter_ids)

    w_min, w_max = min(
        [torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]
    ), max([torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
    h_min, h_max = min(
        [torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]
    ), max([torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

    c_w = (-w_min + w_max) / 2
    c_h = (-h_min + h_max) / 2

    svg_result = os.path.join(experiment_dir, "output-svg", "output.svg")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg_result
    )

    out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes]
    )
    out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes]
    )

    out_c_w = (-out_w_min + out_w_max) / 2
    out_c_h = (-out_h_min + out_h_max) / 2

    scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
    scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

    if scale_canvas_h > scale_canvas_w:
        wsize = int((out_w_max - out_w_min) * scale_canvas_h)
        scale_canvas_w = wsize / (out_w_max - out_w_min)
        shift_w = -out_c_w * scale_canvas_w + c_w
    else:
        hsize = int((out_h_max - out_h_min) * scale_canvas_w)
        scale_canvas_h = hsize / (out_h_max - out_h_min)
        shift_h = -out_c_h * scale_canvas_h + c_h

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h
        if scale_canvas_h > scale_canvas_w:
            p.points[:, 0] = (
                p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
            )
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
        else:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
            p.points[:, 1] = (
                p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h
            )

    for j, s in enumerate(letter_ids):
        shapes_word[s] = shapes[j]

    save_svg.save_svg(
        f"{experiment_dir}/{font}_{word}_{letter}.svg",
        canvas_width,
        canvas_height,
        shapes_word,
        shape_groups_word,
    )

    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes_word, shape_groups_word
    )
    import torch 
    import torch_xla.core.xla_model as xm

    device  = xm.xla_device()
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=device
    ) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    save_image(img, f"{experiment_dir}/{font}_{word}_{letter}.png")


def create_video(num_iter, experiment_dir, video_frame_freq):
    img_array = []
    for ii in range(0, num_iter):
        if ii % video_frame_freq == 0 or ii == num_iter - 1:
            filename = os.path.join(experiment_dir, "video-png", f"iter{ii:04d}.png")
            img = cv2.imread(filename)
            img_array.append(img)

    video_name = os.path.join(experiment_dir, "video.mp4")
    check_and_create_dir(video_name)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (600, 600))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()
