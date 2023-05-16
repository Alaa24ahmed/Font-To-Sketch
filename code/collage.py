import os
import imageio 
import numpy as np
from glob import glob 
from PIL import Image, ImageSequence

if __name__ == "__main__":
    
    path = "/Users/bkhmsi/Desktop/Animal-Words/*.gif"
    save_path = os.path.join(os.path.dirname(path), "collage.gif")


    width, height = 600, 600
    nx, ny = 2, 2
    n_frames = 67
    collage = np.ones((n_frames, width*nx, height*ny)).astype(np.uint8)

    filenames = [p for p in glob(path) if "collage" not in p]
    assert len(filenames) <= nx*ny

    for i in range(nx):
        for j in range(ny):
            image = Image.open(filenames[i*ny+j])
            assert image.is_animated
            for frame_idx in range(image.n_frames):
                image.seek(frame_idx)
                frame = image.convert('L').copy()
                collage[frame_idx, i*width:(i+1)*width,j*height:(j+1)*height] = frame

    imageio.mimsave(save_path, collage)
