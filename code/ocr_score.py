# import lpips
# import torchvision.models as models
# import torchvision
# import wandb
# import torchvision.transforms as transforms
# from ocr.recognition_model import load_model
# from typing import List
# from ocr.processor import load_processor
# import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# import torch.nn.functional as F
# # from ocr.postprocessing.text import truncate_repetitions
# from ocr.settings import settings
# from ocr.schema import OCRResult


from PIL import Image
# from surya.surya.ocr import run_ocr
# from surya.surya.model.detection import segformer
from surya_ocr.surya.model.recognition.model import load_model
from surya_ocr.surya.model.recognition.processor import load_processor


from typing import List
import torch
from PIL import Image
from transformers import GenerationConfig
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
# from surya.surya.postprocessing.math.latex import fix_math, contains_math
# from surya.surya.postprocessing.text import truncate_repetitions
from surya_ocr.surya.settings import settings
from surya_ocr.surya.schema import TextLine, OCRResult


CODE_TO_LANGUAGE = {
    'af': 'afrikaans',
    'am': 'amharic',
    'ar': 'arabic',
    'as': 'assamese',
    'az': 'azerbaijani',
    'be': 'belarusian',
    'bg': 'bulgarian',
    'bn': 'bengali',
    'br': 'breton',
    'bs': 'bosnian',
    'ca': 'catalan',
    'cs': 'czech',
    'cy': 'welsh',
    'da': 'danish',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'eo': 'esperanto',
    'es': 'spanish',
    'et': 'estonian',
    'eu': 'basque',
    'fa': 'persian',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'western frisian',
    'ga': 'irish',
    'gd': 'scottish gaelic',
    'gl': 'galician',
    'gu': 'gujarati',
    'ha': 'hausa',
    'he': 'hebrew',
    'hi': 'hindi',
    'hr': 'croatian',
    'hu': 'hungarian',
    'hy': 'armenian',
    'id': 'indonesian',
    'is': 'icelandic',
    'it': 'italian',
    'ja': 'japanese',
    'jv': 'javanese',
    'ka': 'georgian',
    'kk': 'kazakh',
    'km': 'khmer',
    'kn': 'kannada',
    'ko': 'korean',
    'ku': 'kurdish',
    'ky': 'kyrgyz',
    'la': 'latin',
    'lo': 'lao',
    'lt': 'lithuanian',
    'lv': 'latvian',
    'mg': 'malagasy',
    'mk': 'macedonian',
    'ml': 'malayalam',
    'mn': 'mongolian',
    'mr': 'marathi',
    'ms': 'malay',
    'my': 'burmese',
    'ne': 'nepali',
    'nl': 'dutch',
    'no': 'norwegian',
    'om': 'oromo',
    'or': 'oriya',
    'pa': 'punjabi',
    'pl': 'polish',
    'ps': 'pashto',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sa': 'sanskrit',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'sq': 'albanian',
    'sr': 'serbian',
    'su': 'sundanese',
    'sv': 'swedish',
    'sw': 'swahili',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tl': 'tagalog',
    'tr': 'turkish',
    'ug': 'uyghur',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'zh': 'chinese',
}

LANGUAGE_TO_CODE = {v: k for k, v in CODE_TO_LANGUAGE.items()}

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
                batch_size = 64 # 12GB RAM max
            if settings.TORCH_DEVICE_MODEL == "cuda":
                batch_size = 256
        return batch_size


    def batch_recognition(self, images: List, languages: List[List[str]], model, processor, batch_size=None):
        assert all([isinstance(image, Image.Image) for image in images])
        assert len(images) == len(languages)

        if batch_size is None:
            batch_size = self.get_batch_size()

        output_text = []
        confidences = []

        for i in tqdm(range(0, len(images), batch_size), desc="Recognizing Text"):
            batch_langs = languages[i:i+batch_size]
            has_math = ["_math" in lang for lang in batch_langs]
            batch_images = images[i:i+batch_size]
            batch_images = [image.convert("RGB") for image in batch_images]
            # for img in batch_images:
            #     plt.imshow(img)
            #     plt.axis('off')
            #     plt.show()
                
            model_inputs = processor(text=[""] * len(batch_langs), images=batch_images, lang=batch_langs)

            batch_pixel_values = model_inputs["pixel_values"]
            batch_langs = model_inputs["langs"]
            batch_decoder_input = [[model.config.decoder_start_token_id] + lang for lang in batch_langs]

            batch_langs = torch.from_numpy(np.array(batch_langs, dtype=np.int64)).to(model.device)
            batch_pixel_values = torch.tensor(np.array(batch_pixel_values), dtype=model.dtype).to(model.device)
            batch_decoder_input = torch.from_numpy(np.array(batch_decoder_input, dtype=np.int64)).to(model.device)

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
                scores = return_dict["scores"] # Scores is a tuple, one per new sequence position.  Each tuple element is bs x vocab_size
                print(len(scores))
                sequence_scores = torch.zeros(generated_ids.shape[0])
                sequence_lens = torch.where(
                    generated_ids > processor.tokenizer.eos_id,
                    torch.ones_like(generated_ids),
                    torch.zeros_like(generated_ids)
                ).sum(axis=-1).cpu()
                prefix_len = generated_ids.shape[1] - len(scores) # Length of passed in tokens (bos, langs)
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
            # Postprocess to fix LaTeX output (add $$ signs, etc)
            # detected_text = [fix_math(text) if math and contains_math(text) else text for text, math in zip(detected_text, has_math)]
            output_text.extend(detected_text)
            confidences.extend(sequence_scores.tolist())

        return output_text, confidences


    def recognize_image(self, image: Image.Image, rec_model, rec_processor, batch_size=None) -> List[OCRResult]:
        all_langs=[[LANGUAGE_TO_CODE[self.cfg.script]]]
        print(all_langs)
        return self.batch_recognition([image], all_langs, self.rec_model, self.rec_processor, batch_size=batch_size)

    def get_score(self, image):
        # self.image = image.permute(2, 0, 1).unsqueeze(0)
        ocr_score_res = self.recognize_image(image, self.rec_model, self.rec_processor)
        detected_text, confidence_score = ocr_score_res
        print(detected_text)
        return detected_text, confidence_score
        # score = confidence_score - 
