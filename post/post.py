import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
import matplotlib.pyplot as plt

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16, resume_download=True
)
pipe.to("cuda")


# url = "https://i.imgur.com/y7EWJUx.png"
# init_image = Image.open(requests.get(url, stream=True).raw)
# img_path = "15_قرد_[0, 1, 2].png"
# img_path = "16_زرافة_[2].png"
# img_path = "01_هي_[0, 1].png"
# img_path = "01_يوغا_[0, 1, 2, 3].png"
# img_path= "Untitled (3).png"
# img_path = "06_فهد_1_dot_loss_0_content_loss0_angels_loss0.5_seed_42_levelOfcc_1_sigma_60_leopard.png"
# img_path = 'C:\\Users\\samal\\Downloads\\06_ضفدع_[2] (1) (1).png'
# img_path = 'C:\\Users\\samal\\Downloads\\16_زرافة_[2] (2).png'
# img_path = 'C:\\Users\\samal\\Downloads\\15_ثعلب_[1, 2] (3).png'
# img_path = 'D:\\UNI\\GRADUATION\\us\\Font-To-Sketch\\output\\kindness.png'
# img_path = 'D:\\UNI\\GRADUATION\\RESULTS\\us\\Italy\\output.png'
# img_path = "D:\\UNI\\GRADUATION\\RESULTS\\us\\Palestine\\output (19).png"
# img_path ='D:\\UNI\\GRADUATION\\RESULTS\\us\\Italy\\15_إيطاليا_[2, 3, 4] (1).png'
# # Convert base image to RGBA if it's not already
img_path="/home/ahmed.sharshar/Desktop/Alaa/Font-To-Sketch/output_2/arabic/06_طائر_23_seed_42_ocr_loss_1/06_طائر_[2, 3].png"
init_image = Image.open(img_path).convert("RGB")

# if init_image.mode != 'RGBA':
#     base_image = init_image.convert('RGBA')
# # img_path = "input_post.png"

# prompt = "minimal flat 2d vector. " + "Monkey"  + " with monkey colors trending on artstation"
# n_propmt = "bad, deformed, ugly, bad anotomy"

# prompt = "A vibrant, colorful , minimalist 2D vector illustration of a giraffe, using trending colors from ArtStation."
concept = "A bird"
# prompt = "A vibrant, minimalist 2D vector illustration of a she, using colors typically associated with shes objects."
prompt = "A vibrant, minimalist 2D vector illustration of "+concept+", add colors typically associated with "+ concept+" objects."
n_prompt = "Deformities, ugliness, and incorrect anatomy"
strength = 0.5
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=strength).images[0]
#image = pipe(prompt=prompt, image=init_image,  strength=0.5).images[0]

save_path = f"/home/ahmed.sharshar/Desktop/Alaa/Font-To-Sketch/post/{concept}.png"
image.save(save_path)
# plt.imshow(image)


# # Open the base image
# base_image = Image.open('16_زرافة_[2].png')

# # Convert base image to RGBA if it's not already
# if base_image.mode != 'RGBA':
#     base_image = base_image.convert('RGBA')

# base_image.save('base_image.png')
# # Open the image to overlay
# overlay_image = Image.open('output_post.png')

# # Resize or position overlay_image as needed
# overlay_image = overlay_image.resize(base_image.size, Image.Resampling.LANCZOS)

# # Ensure overlay_image is in RGBA mode
# # if overlay_image.mode != 'RGBA':
# #     overlay_image = overlay_image.convert('RGBA')


# # Adjust the transparency of the overlay image
# # Create an alpha mask
# alpha_mask = Image.new("L", overlay_image.size, 50)  # 128 is semi-transparent
# overlay_image.putalpha(alpha_mask)

# # Paste overlay_image on top of base_image
# base_image.paste(overlay_image, (0, 0), overlay_image)

# # Save the result
# base_image.save('combined_image.png')

