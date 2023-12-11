from typing import Mapping

import clip
import pydiffvg
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from ttf import extract_svg_paths
from easydict import EasyDict as edict

def process_image_to_pytorch(batch_size, image):
    image = image.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
    image = image.repeat(batch_size, 1, 1, 1)
    return image

pydiffvg.set_use_gpu(torch.cuda.is_available())
device = pydiffvg.get_device()

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

class CLipDrawLoss():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model, self.preprocess = clip.load('ViT-B/32', device, jit=False)

    def get_loss(self):
        use_normalized_clip = True

        augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
         ])

        if use_normalized_clip:
            augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        h, w = self.cfg.render_size, self.cfg.render_size
        shapes, shape_groups, parameters = init_shapes(svg_path=self.cfg.target, trainable=self.cfg.trainable)
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(w, h, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = process_image_to_pytorch(self.cfg.batch_size, img)
        # img = img.squeeze(0).permute(1, 2, 0)


        # img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # img = img[:, :, :3]
        # img = img.unsqueeze(0)
        # img = img.permute(0, 3, 1, 2) # NHWC -> NCHW


        text_input = clip.tokenize(self.cfg.caption).to(device)
        text_features = self.model.encode_text(text_input)

        loss = 0
        self.NUM_AUGS = 10
        img_augs = []
        for n in range(self.NUM_AUGS):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        image_features = self.model.encode_image(im_batch)
        for n in range(self.NUM_AUGS):
            loss += torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
        # im_batch = torch.cat([img])
        # image_features = self.model.encode_image(im_batch)
        # loss = torch.cosine_similarity(text_features, image_features[0:1], dim=1)
        
        return loss

    def get_loss_per_region(self):
        init_path = f"code/data/init"
        svg_path = f"{init_path}/{self.cfg.experiment_name}"
        svg_path = svg_path.replace(" ", "_")
        losses = {}
        for mx in range(0, len(self.cfg.word)+1):
            for mn in  range(0,mx):
                letters = [i for i in range(mn, mx)]
                print(letters)
                extract_svg_paths(svg_path, letters, self.cfg.script)
                loss = self.get_loss()
                loss = loss.to(torch.float32)  # Convert to float32 if needed
                loss_float = loss.item()
                letter_in_word = "".join([self.cfg.word[i] for i in letters])
                losses[letter_in_word] = (abs(loss_float)/self.NUM_AUGS)*100
        print(self.cfg.word)
        sorted_dict = dict(sorted(losses.items(), key=lambda item: item[1], reverse=True))
        for letters, loss in sorted_dict.items():
            print(f"loss for {letters} is {loss}")


        

# CLipDrawLoss(device)