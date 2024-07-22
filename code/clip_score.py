# from transformers import CLIPProcessor, CLIPModel
# import torch
# from torchvision import transforms

# class CLipScoring:
#     def __init__(self, cfg, device):
#         self.cfg = cfg
#         self.device = device
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
#     def process_image_to_pytorch(self, image, batch_size):
#         image = image.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
#         image = image.repeat(batch_size, 1, 1, 1)
#         return image

#     def get_loss(self, img, caption, batch_size=1):
#         augment_trans = transforms.Compose([
#             transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
#             transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
#             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#         ])

#         # img = self.process_image_to_pytorch(image, batch_size).to(self.device)

#         text_input = self.processor(text=[caption], return_tensors="pt").to(self.device)
#         text_features = self.model.get_text_features(**text_input)

#         loss = 0
#         NUM_AUGS = 1
#         img_augs = [augment_trans(img) for _ in range(NUM_AUGS)]
#         im_batch = torch.cat(img_augs)
#         image_features = self.model.get_image_features(pixel_values=im_batch)

#         for n in range(NUM_AUGS):
#             loss += torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)

#         return (loss / NUM_AUGS).item()  # Average the loss over the augmentations and convert to a scalar

#         # return loss / NUM_AUGS  # Average the loss over the augmentations




# from transformers import CLIPProcessor, CLIPModel
# import torch
# from torchvision import transforms

# class CLipScoring:
#     def __init__(self, cfg, device):
#         self.cfg = cfg
#         self.device = device
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#     def get_loss(self, img, caption, batch_size=1):
#         # Normalize the image as per CLIP's requirements
#         preprocess = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
#                                  (0.26862954, 0.26130258, 0.27577711))
#         ])
        
#         # Ensure img is in NCHW format and has correct shape
#         # img = preprocess(img).unsqueeze.to(self.device)

#         text_input = self.processor(text=[caption], return_tensors="pt").to(self.device)
#         text_features = self.model.get_text_features(**text_input)
#         image_features = self.model.get_image_features(pixel_values=img)

#         # Calculate cosine similarity between text and image features
#         loss = torch.cosine_similarity(text_features, image_features, dim=1)
#         return loss.item()  # Convert to scalar





from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms

class CLipScoring:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def process_image_to_pytorch(self, image, batch_size):
        image = image.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        image = image.repeat(batch_size, 1, 1, 1)
        return image

    def get_loss(self, img, caption, batch_size=1):
        use_normalized_clip = True

        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
        ])

        if use_normalized_clip:
            augment_trans = transforms.Compose([
                transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
                transforms.Normalize((0.5,), (0.5,))  # Assuming mean and std for grayscale
                # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

        # img = self.process_image_to_pytorch(image, batch_size).to(self.device)

        text_input = self.processor(text=[caption], return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**text_input)

        loss = 0
        NUM_AUGS = 10
        img_augs = [augment_trans(img) for _ in range(NUM_AUGS)]
        im_batch = torch.cat(img_augs)
        image_features = self.model.get_image_features(pixel_values=im_batch)

        for n in range(NUM_AUGS):
            loss += torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)

        return (loss / NUM_AUGS).item()  # Average the loss over the augmentations and convert to a scalar

        # return loss / NUM_AUGS  # Average the loss over the augmentations







