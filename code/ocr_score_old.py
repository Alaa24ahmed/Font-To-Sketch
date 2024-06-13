import lpips
import torchvision.models as models
import torchvision
import wandb
import torchvision.transforms as transforms
from ocr.recognition_model import load_model
from typing import List
from ocr.processor import load_processor
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
# from ocr.postprocessing.text import truncate_repetitions
from ocr.settings import settings
from ocr.schema import OCRResult


class OcrScoring(nn.Module):    
    def __init__(self, cfg, device): 
        super(OcrScoring, self).__init__()
        self.cfg = cfg
        self.device = device
        self.rec_model, self.rec_processor = load_model(), load_processor()
        self.rec_model = self.rec_model.float()
        
    def get_batch_size(self):
        batch_size = settings.RECOGNITION_BATCH_SIZE
        if batch_size is None:
            batch_size = 32
            if settings.TORCH_DEVICE_MODEL == "mps":
                batch_size = 64  # 12GB RAM max
            if settings.TORCH_DEVICE_MODEL == "cuda":
                batch_size = 256
        return batch_size

    def batch_recognition(self, images: List[Image.Image], languages: List[List[str]], model, processor, batch_size=None):
        # assert all([isinstance(image, Image.Image) for image in images])
        assert all([isinstance(image, torch.Tensor) for image in images]), "All items in images must be of type torch.Tensor"
        assert len(images) == len(languages)

        if batch_size is None:
            batch_size = self.get_batch_size()

        output_text = []
        confidences = []

        for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
            batch_langs = languages[i:i+batch_size]
            has_math = ["_math" in lang for lang in batch_langs]
            batch_images = images[i:i+batch_size]
            # batch_images = [image.convert("RGB") for image in batch_images]
            batch_images = [image.to(model.device).type(model.dtype) for image in batch_images]
            batch_images = torch.stack(batch_images)
            model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

            batch_pixel_values = model_inputs["pixel_values"]
            batch_langs = model_inputs["langs"]
            batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

            batch_langs = torch.tensor(batch_langs, dtype=torch.int64).to(model.device)
            batch_pixel_values = torch.tensor(batch_pixel_values, dtype=model.dtype).to(model.device)
            batch_decoder_input = torch.tensor(batch_decoder_input, dtype=torch.int64).to(model.device)

            with torch.inference_mode():
                return_dict = model.generate(
                    pixel_values=batch_pixel_values,
                    decoder_input_ids=batch_decoder_input,
                    decoder_langs=batch_langs,
                    eos_token_id=processor.tokenizer.eos_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    max_new_tokens=settings.RECOGNITION_MAX_TOKENS,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                generated_ids = return_dict["sequences"]

                # Find confidence scores
                scores = return_dict["scores"]
                sequence_scores = torch.zeros(generated_ids.shape[0], device=model.device)
                sequence_lens = torch.where(
                    generated_ids > processor.tokenizer.eos_id,
                    torch.ones_like(generated_ids),
                    torch.zeros_like(generated_ids)
                ).sum(axis=-1).cpu()
                prefix_len = generated_ids.shape[1] - len(scores)  # Length of passed in tokens (bos, langs)
                for token_idx, score in enumerate(scores):
                    probs = F.softmax(score, dim=-1)
                    max_probs = torch.max(probs, dim=-1).values
                    max_probs = torch.where(
                        generated_ids[:, token_idx + prefix_len] <= processor.tokenizer.eos_id,
                        torch.zeros_like(max_probs),
                        max_probs
                    ).cpu()
                    sequence_scores += max_probs
                sequence_scores /= sequence_lens
            detected_text = processor.tokenizer.batch_decode(generated_ids)

            # detected_text = [truncate_repetitions(dt) for dt in detected_text]
            output_text.extend(detected_text)
            confidences.extend(sequence_scores.tolist())

        return output_text, confidences, return_dict

    def recognize_image(self, image: Image.Image, rec_model, rec_processor, batch_size=None) -> List[OCRResult]:
        all_langs = [["ar"]] 
        return self.batch_recognition([image], all_langs, self.rec_model, self.rec_processor, batch_size=batch_size)

    def get_score(self, image):
        # self.image = image.permute(2, 0, 1).unsqueeze(0)
        return self.recognize_image(image, self.rec_model, self.rec_processor)

# import lpips
# import torchvision.models as models
# import torchvision
# import wandb
# import torchvision.transforms as transforms
# from ocr.recognition_model import load_model
# from typing import List
# from ocr.processor import load_processor
# import torch
# import torch.nn as nn

# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# import torch.nn.functional as F
# # from ocr.postprocessing.text import truncate_repetitions
# from ocr.settings import settings
# from ocr.schema import OCRResult


# class OcrScoring(nn.Module):    
#     def __init__(self , cfg, device): 
#         super(OcrScoring, self).__init__()
#         self.cfg = cfg
#         self.device = device
#         self.rec_model, self.rec_processor = load_model(), load_processor()
#         self.rec_model = self.rec_model.float() 
        
#     ## recognize image code 
#     # def process_image_to_pytorch(self, image, batch_size):
#     #     image = image.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
#     #     image = image.repeat(batch_size, 1, 1, 1)
#     #     return image 
    
#     def get_batch_size(self):
#         batch_size = settings.RECOGNITION_BATCH_SIZE
#         if batch_size is None:
#             batch_size = 32
#             if settings.TORCH_DEVICE_MODEL == "mps":
#                 batch_size = 64  # 12GB RAM max
#             if settings.TORCH_DEVICE_MODEL == "cuda":
#                 batch_size = 256
#         return batch_size


#     def batch_recognition(self, images: List, languages: List[List[str]], model, processor, batch_size=None):
#         assert all([isinstance(image, Image.Image) for image in images])
#         assert len(images) == len(languages)

#         if batch_size is None:
#             batch_size = self.get_batch_size()

#         output_text = []
#         confidences = []

#         for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
#             batch_langs = languages[i:i+batch_size]
#             has_math = ["_math" in lang for lang in batch_langs]
#             batch_images = images[i:i+batch_size]
#             batch_images = [image.convert("RGB") for image in batch_images]

#             model_inputs = self.rec.processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

#             batch_pixel_values = model_inputs["pixel_values"]
#             batch_langs = model_inputs["langs"]
#             batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

#             batch_langs = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
#             batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
#             batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

#             with torch.inference_mode():
#                 return_dict = model.generate(
#                     pixel_values=batch_pixel_values,
#                     decoder_input_ids=batch_decoder_input,
#                     decoder_langs=batch_langs,
#                     eos_token_id=self.rec.processor.tokenizer.eos_id,
#                     pad_token_id=self.rec.processor.tokenizer.pad_token_id,
#                     max_new_tokens=settings.RECOGNITION_MAX_TOKENS,
#                     output_scores=True,
#                     return_dict_in_generate=True
#                 )
#                 generated_ids = return_dict["sequences"]

#                 # Find confidence scores
#                 scores = return_dict["scores"] # Scores is a tuple, one per new sequence position.  Each tuple element is bs x vocab_size
#                 print(len(scores))
#                 sequence_scores = torch.zeros(generated_ids.shape[0])
#                 sequence_lens = torch.where(
#                     generated_ids > self.rec.processor.tokenizer.eos_id,
#                     torch.ones_like(generated_ids),
#                     torch.zeros_like(generated_ids)
#                 ).sum(axis=-1).cpu()
#                 prefix_len = generated_ids.shape[1] - len(scores) # Length of passed in tokens (bos, langs)
#                 for token_idx, score in enumerate(scores):
#                     probs = F.softmax(score, dim=-1)
#                     max_probs = torch.max(probs, dim=-1).values
#                     max_probs = torch.where(
#                         generated_ids[:, token_idx + prefix_len] <= self.rec.processor.tokenizer.eos_id,
#                         torch.zeros_like(max_probs),
#                         max_probs
#                     ).cpu()
#                     sequence_scores += max_probs
#                 sequence_scores /= sequence_lens
#             detected_text = self.self.rec.processor.tokenizer.batch_decode(generated_ids)
        
#             # detected_text = [truncate_repetitions(dt) for dt in detected_text]
#             # Postprocess to fix LaTeX output (add $$ signs, etc)
#             # detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
#             output_text.extend(detected_text)
#             confidences.extend(sequence_scores.tolist())

#         return output_text, confidences, return_dict


#     def recognize_image(self, image: Image.Image, rec_model, rec_processor, batch_size=None) -> List[OCRResult]:
#         all_langs = [["ar"]] 
#         return self.batch_recognition([image], all_langs, self.rec_model, self.rec_processor, batch_size=batch_size)


#     def get_score(self, image):
#         # self.image = image.permute(2, 0, 1).unsqueeze(0)
#         return self.recognize_image(image, self.rec_model, self.rec_processor)

#############################

    # def batch_recognition(self, images: List[Image.Image], languages: List[List[str]], model, processor, batch_size=None):
    #     # assert all(isinstance(image, Image.Image) for image in images)
    #     # assert len(images) == len(languages)

    #     if batch_size is None:
    #         batch_size = self.get_batch_size()

    #     output_text = []
    #     confidences = []
        

    #     for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
    #         batch_langs = languages[i:i + batch_size]
    #         batch_images = images[i:i + batch_size]
    #         batch_images = [image.convert("RGB") for image in batch_images]

    #         model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

    #         batch_pixel_values = model_inputs["pixel_values"]
    #         batch_langs = model_inputs["langs"]
    #         batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

    #         batch_langs = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
    #         batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
    #         batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

    #         with torch.inference_mode():
    #             return_dict = model.generate(
    #                 pixel_values=batch_pixel_values,
    #                 decoder_input_ids=batch_decoder_input,
    #                 decoder_langs=batch_langs,
    #                 eos_token_id=processor.tokenizer.eos_id,
    #                 pad_token_id=processor.tokenizer.pad_token_id,
    #                 max_new_tokens=settings.RECOGNITION_MAX_TOKENS,
    #                 output_scores=True,
    #                 return_dict_in_generate=True
    #             )
    #             generated_ids = return_dict["sequences"]

    #             # Find confidence scores
    #             scores = return_dict["scores"]
    #             sequence_scores = torch.zeros(generated_ids.shape[0])
    #             sequence_lens = torch.where(
    #                 generated_ids > processor.tokenizer.eos_id,
    #                 torch.ones_like(generated_ids),
    #                 torch.zeros_like(generated_ids)
    #             ).sum(axis=-1).cpu()
    #             prefix_len = generated_ids.shape[1] - len(scores)  # Length of passed in tokens (bos, langs)
    #             for token_idx, score in enumerate(scores):
    #                 probs = F.softmax(score, dim=-1)
    #                 max_probs = torch.max(probs, dim=-1).values
    #                 max_probs = torch.where(
    #                     generated_ids[:, token_idx + prefix_len] <= processor.tokenizer.eos_id,
    #                     torch.zeros_like(max_probs),
    #                     max_probs
    #                 ).cpu()
    #                 sequence_scores += max_probs
    #             sequence_scores /= sequence_lens
    #         detected_text = processor.tokenizer.batch_decode(generated_ids)
    #         # detected_text = [truncate_repetitions(dt) for dt in detected_text]
    #         output_text.extend(detected_text)
    #         confidences.extend(sequence_scores.tolist())

    #     return output_text, confidences, return_dict
    
    
    
    # def batch_recognition(self, image_tensor: torch.Tensor, languages: List[str], model, processor, batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.get_batch_size()

    #     output_text = []
    #     confidences = []

    #     # Ensure image tensor is in the correct device and dtype
    #     image_tensor = image_tensor.to(model.device).type(model.dtype)

    #     model_inputs = processor(text=[""] * len(languages), images=[image_tensor], lang=[languages])
        
    #     pixel_values = model_inputs["pixel_values"]
    #     langs = model_inputs["langs"]
    #     decoder_input = [[model.config.decoder_start_token_id] + lang for lang in langs]

    #     langs = torch.tensor(langs, dtype=torch.int64).to(model.device)
    #     pixel_values = torch.tensor(pixel_values, dtype=model.dtype).to(model.device)
    #     decoder_input = torch.tensor(decoder_input, dtype=torch.int64).to(model.device)

    #     with torch.inference_mode():
    #         return_dict = model.generate(
    #             pixel_values=pixel_values,
    #             decoder_input_ids=decoder_input,
    #             decoder_langs=langs,
    #             eos_token_id=processor.tokenizer.eos_id,
    #             pad_token_id=processor.tokenizer.pad_token_id,
    #             max_new_tokens=settings.RECOGNITION_MAX_TOKENS,
    #             output_scores=True,
    #             return_dict_in_generate=True
    #         )
    #         generated_ids = return_dict["sequences"]

    #         # Find confidence scores
    #         scores = return_dict["scores"]
    #         sequence_scores = torch.zeros(generated_ids.shape[0], device=model.device)
    #         sequence_lens = torch.where(
    #             generated_ids > processor.tokenizer.eos_id,
    #             torch.ones_like(generated_ids),
    #             torch.zeros_like(generated_ids)
    #         ).sum(axis=-1).cpu()
    #         prefix_len = generated_ids.shape[1] - len(scores)  # Length of passed in tokens (bos, langs)
    #         for token_idx, score in enumerate(scores):
    #             probs = F.softmax(score, dim=-1)
    #             max_probs = torch.max(probs, dim=-1).values
    #             max_probs = torch.where(
    #                 generated_ids[:, token_idx + prefix_len] <= processor.tokenizer.eos_id,
    #                 torch.zeros_like(max_probs),
    #                 max_probs
    #             ).cpu()
    #             sequence_scores += max_probs
    #         sequence_scores /= sequence_lens

    #     detected_text = processor.tokenizer.batch_decode(generated_ids)
    #     output_text.extend(detected_text)
    #     confidences.extend(sequence_scores.tolist())

        # return output_text, confidences, return_dict
    # def recognize_image(self, image: Image.Image, rec_model, rec_processor, batch_size=None) -> List[OCRResult]:
    #     all_langs = [["ar"]]
    #     output_text, confidences, return_dict = self.batch_recognition([image], all_langs, rec_model, rec_processor, batch_size=batch_size)
    #     return output_text, confidences


    


