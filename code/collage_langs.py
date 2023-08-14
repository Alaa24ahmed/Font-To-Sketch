import os
import imageio 
import numpy as np
from glob import glob 
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def create_image(size, text, font):
    W, H = size
    image = Image.new('L', size, 255)
    draw = ImageDraw.Draw(image)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    draw.text(((W-w)/2, (H-h)/2), text, font=font, fill=0)
    return image

if __name__ == "__main__":

    languages = ['arabic', 'greek', 'chinese', 'russian', 'tamil', 'english']
    scripts = ['arabic', 'greek', 'chinese', 'cyrillic', 'tamil', 'latin']
    concepts = ['panda', 'car', 'music', 'bird', 'star', 'cloud']

    languages = ['arabic', 'greek', 'chinese', 'russian', 'tamil']
    scripts = ['arabic', 'greek', 'chinese', 'cyrillic', 'tamil']
    concepts = ['car']

    sizew, sizeh = 160, 160
    width, height = 130, 130
    diff_offset = int((sizew-width) / 2)
    n_frames = 67
    freeze = 5
    text_height = 30
    nx, ny = len(concepts), len(scripts)
    collage = np.ones((n_frames*2+freeze-1, text_height+width*nx, height*ny)).astype(np.uint8)*255
    
    savepath = "../images/languages_car.gif"
    dirpath = "../examples/concepts"

    font = ImageFont.truetype('data/fonts/latin/Roboto-Regular.ttf', 25)
    for i in tqdm(range(ny)):
        background = create_image((width, text_height), languages[i].capitalize(), font)
        lang_image = np.asarray(background)
        for idx in range(n_frames*2+freeze-1):
            collage[idx, :text_height, i*width:(i+1)*width] = lang_image
    
    for i, concept in tqdm(enumerate(concepts), total=len(concepts)):
        for j, script in enumerate(scripts):
            filepath = os.path.join(dirpath, concept, f"{script}.gif")
            image = Image.open(filepath)
            assert image.is_animated
            image.seek(0)
            frame = image.convert('L').copy()
            frame = frame.resize((sizew,sizeh))
            for idx in range(freeze):
                collage[idx, text_height+i*width:text_height+(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)[diff_offset:sizew-diff_offset,diff_offset:sizew-diff_offset]

            for frame_idx in range(n_frames):
                image.seek(frame_idx)
                frame = image.convert('L').copy()
                frame = frame.resize((sizew,sizeh))
                collage[idx, text_height+i*width:text_height+(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)[diff_offset:sizew-diff_offset,diff_offset:sizew-diff_offset]
                idx += 1

            for frame_idx in reversed(range(n_frames)):
                image.seek(frame_idx)
                frame = image.convert('L').copy()
                frame = frame.resize((sizew,sizeh))
                collage[idx, text_height+i*width:text_height+(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)[diff_offset:sizew-diff_offset,diff_offset:sizew-diff_offset]
                idx += 1

    imageio.mimsave(savepath, collage)