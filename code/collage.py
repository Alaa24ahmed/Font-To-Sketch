import os
import imageio 
import numpy as np
from glob import glob 
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__":
    
    path = "/Users/bkhmsi/Desktop/Animal-Words/*.gif"
    save_path = os.path.join(os.path.dirname(path), "collage_loop_25_2.gif")


    width, height = 400, 400
    width, height = 100, 100
    nx, ny = 5, 5
    n_frames = 67
    collage = np.ones((n_frames*2, width*nx, height*ny)).astype(np.uint8)*255

    filenames = [p for p in glob(path) if os.path.basename(p)[:-4] not in ["palestine", "amin", "collage", "collage_loop_25", "collage_loop_7", "collage_1d"]]
    print(f"> {len(filenames)} Files Found")

    f_filenames = filenames
    filter = ["horse.gif", "giraffe.gif", "duck.gif", "turtle.gif", "camel.gif", "octopus.gif", "shark.gif"]
    # f_filenames = []
    # for file in filenames:
    #     basename = os.path.basename(file)
    #     if basename in filter:
    #         f_filenames += [file]

    assert nx*ny <= len(f_filenames)

    for i in range(nx):
        for j in tqdm(range(ny)):
            image = Image.open(f_filenames[i*ny+j])
            assert image.is_animated
            idx = 0
            for frame_idx in range(n_frames):
                image.seek(frame_idx)
                frame = image.convert('L').copy()
                frame = frame.resize((width,height))
                collage[idx, i*width:(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)
                idx += 1

            for frame_idx in reversed(range(n_frames)):
                image.seek(frame_idx)
                frame = image.convert('L').copy()
                frame = frame.resize((width,height))
                collage[idx, i*width:(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)
                idx += 1


    imageio.mimsave(save_path, collage)
