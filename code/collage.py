import os
import imageio 
import numpy as np
from glob import glob 
from PIL import Image, ImageSequence

if __name__ == "__main__":
    
    path = "/Users/bkhmsi/Desktop/Animal-Words/*.gif"
    save_path = os.path.join(os.path.dirname(path), "collage_1d.gif")


    width, height = 400, 400
    nx, ny = 1, 7
    n_frames = 67
    collage = np.ones((n_frames+10, width*nx, height*ny)).astype(np.uint8)

    filenames = [p for p in glob(path) if os.path.basename(p)[:-4] not in ["palestine", "amin", "collage"]]
    print(f"> {len(filenames)} Files Found")

    filter = ["horse.gif", "giraffe.gif", "duck.gif", "turtle.gif", "camel.gif", "octopus.gif", "shark.gif"]
    f_filenames = []
    for file in filenames:
        basename = os.path.basename(file)
        if basename in filter:
            f_filenames += [file]



    assert nx*ny <= len(f_filenames)

    for i in range(nx):
        for j in range(ny):
            image = Image.open(f_filenames[i*ny+j])
            assert image.is_animated
            idx = 0
            for frame_idx in range(image.n_frames):
                image.seek(frame_idx)
                frame = image.convert('L').copy()
                if frame_idx == 0 or frame_idx == image.n_frames-1:
                    for _ in range(5): 
                        collage[idx, i*width:(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)[100:500, 100:500]
                        idx += 1
                else:
                    collage[idx, i*width:(i+1)*width,j*height:(j+1)*height] = np.asarray(frame)[100:500, 100:500]
                    idx += 1

    imageio.mimsave(save_path, collage)
