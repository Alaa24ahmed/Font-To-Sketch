import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from diffusers import StableDiffusionPipeline

import torch.nn.functional as F
import torchvision.models as models
import torchvision


class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg.diffusion.model, torch_dtype=torch.float16, use_auth_token=cfg.token
        )
        self.pipe = self.pipe.to(self.device)
        self.img_init = None
        self.latent_img_init = None
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(
            self.cfg.caption,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input = self.pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
            uncond_embeddings = self.pipe.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
        
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(
            self.cfg.batch_size, 0
        )
        del self.pipe.tokenizer
        del self.pipe.text_encoder

    def set_image_init(self, img_init):
        img_init = img_init * 2.0 - 1.0
        with torch.cuda.amp.autocast():
            self.latent_img_init = self.pipe.vae.encode(img_init).latent_dist.sample()
        self.latent_img_init = (
            0.18215 * self.latent_img_init
        )  # scaling_factor * init_latents
        # make latent_img_init doesn't require grad
        self.latent_img_init = self.latent_img_init.detach()

    def forward(self, x_aug):
        sds_loss = 0

        # generate latent_img_init
        if self.latent_img_init is None:
            raise ValueError("Image init is None")

        # encode rendered image
        x = x_aug * 2.0 - 1.0
        with torch.cuda.amp.autocast():
            unscaled_latent_img = self.pipe.vae.encode(x).latent_dist.sample()
        current_latent_img = (
            0.18215 * unscaled_latent_img
        )  # scaling_factor * init_latents

        with torch.inference_mode():
            self.eval()
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, self.cfg.diffusion.timesteps)
                - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(current_latent_img.shape[0],),
                device=self.device,
                dtype=torch.long,
            )

            # add noise
            eps = torch.randn_like(current_latent_img)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(
                current_latent_img, eps, timestep
            )

            # denoise
            z_in = torch.cat(
                [noised_latent_zt] * 2
            )  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = (
                    self.pipe.unet(
                        z_in, timestep, encoder_hidden_states=self.text_embeddings
                    )
                    .sample.float()
                    .chunk(2)
                )

            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (
                eps_t - eps_t_uncond
            )

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = (
                self.alphas[timestep] ** 0.5 * self.sigmas[timestep] * (eps_t - eps)
            )
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * current_latent_img.clone()
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        print(f"sds_loss: {sds_loss}")

        if self.cfg.use_dot_product_loss:
            init_im_loss = self.latent_img_init.clone() * current_latent_img.clone()
            init_im_loss = init_im_loss.sum(1).mean() * self.cfg.dot_product_loss_weight
            sds_loss = sds_loss - init_im_loss

        return sds_loss


class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(
            kernel_size=(
                cfg.loss.tone.pixel_dist_kernel_blur,
                cfg.loss.tone.pixel_dist_kernel_blur,
            ),
            sigma=(cfg.loss.tone.pixel_dist_sigma),
        )

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)

    def get_scheduler(self, step=None):
        if step is not None:
            # return 100
            return self.dist_loss_weight * np.exp(-(1 / 5) * ((step - 300) / (20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        self.eval()
        return self.mse_loss(
            self.init_blurred.detach(), blurred_cur
        ) * self.get_scheduler(step)


class ConformalLoss:
    def __init__(
        self,
        cfg,
        parameters: EasyDict,
        device: torch.device,
        target_letters: str,
        shape_groups,
    ):


        self.parameters = parameters
        self.target_letters = (
            target_letters
            if cfg.operation_mode == 1
            else list(range(len(shape_groups)))
        )
        print(f"self.target_letters: {self.target_letters}")
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [
            torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))
        ]

        with torch.no_grad():
            self.angles = []
            self.reset()

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i].to("cpu")]
            triangles_roll_a = points[self.faces_roll_a[i].to("cpu")]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum("ned,ned->ne", edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letters):
            print(f"group: {group} ,\n l: {l}")
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                print(f"letter_inds: {letter_inds}")
                print(f"letter_to_insert: {letter_to_insert}")
                print(f"l: {l}")
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        num_shapes = 0
        for j, c in enumerate(self.target_letters):
            points_np = [
                self.parameters.point[i].clone().detach().cpu().numpy()
                for i in range(len(self.parameters.point))
            ]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind + 1 : end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array(
                [poly.contains(Point(points_np[face].mean(0))) for face in faces],
                dtype=np.bool_,
            )
            faces_.append(
                torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64)
            )
            num_shapes += shapes_per_letter
            # if num_shapes >= len(self.target_letters):
            #     break
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            face_loss = nnf.mse_loss(angles[i], self.angles[i])
            loss_angles += face_loss
        return loss_angles


class ContentLoss(nn.Module):
    """
    Calculate content loss between current augmented image and base image.
    Uses Pretrained VGG19 model to calculate content loss.
    Refer here for implementation https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_computer-vision/neural-style.ipynb#scrollTo=283f5e51
    """

    def __init__(self, cfg, base_image):
        super(ContentLoss, self).__init__()
        self.cfg = cfg
        self.vgg_model = models.vgg19(pretrained=True).features.eval()
        self.content_layer = 25  # layer 25 in vgg19 is conv4 used for content loss
        self.initalize_base_image_features(base_image)

    def initalize_base_image_features(self, base_image):
        self.base_image_features = self.get_content_features(self.normalize(base_image))
        self.base_image_features = self.base_image_features.detach()

    def get_content_features(self, current_image):
        x = current_image
        for index, layer in enumerate(self.vgg_model.children()):
            x = layer(x)
            if index == self.content_layer:
                return x.detach()

    def normalize(self, image):
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        mean = torch.tensor(cnn_normalization_mean).view(-1, 1, 1)
        std = torch.tensor(cnn_normalization_std).view(-1, 1, 1)
        image = image.to("cpu")
        return (image - mean) / std

    def forward(self, current_image):
        current_image_features = self.get_content_features(
            self.normalize(current_image)
        )
        return self.cfg.content_loss_weight * F.mse_loss(current_image_features, self.base_image_features)
