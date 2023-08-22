'''
Make a grid illustrating the effects of different values of the parameters on the generated image.

Input:
    folder: the folder containing sub-folders with the different parameter values, inside which are the image
    output: the folder to save the grid 
'''
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def make_grid(folder, output):
    # get the sub-folders
    sub_folders = os.listdir(folder)
    folder_name = folder.split('/')[-1]
    sub_folders = [float(sub_folder) for sub_folder in sub_folders]
    sub_folders.sort()
    sub_folders = [str(sub_folder) for sub_folder in sub_folders]
    # get the images
    images = []
    values = []
    for sub_folder in sub_folders:
        # check if it is a folder

        if not os.path.isdir(os.path.join(folder, sub_folder)):
            continue
        sub_folder_path = os.path.join(folder, sub_folder)
        image_path = os.path.join(sub_folder_path, 'output-png/output.png')
        image = Image.open(image_path)
        images.append(image)
        values.append(sub_folder)
    # make the grid and write the value parameter name and values on the images
    # use matplotlib to plot the images
    fig = plt.figure(figsize=(20, 5))
    columns = len(images)
    rows = 1
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.axis('off')
        plt.title(f'{folder_name}: {values[i-1]}')
    plt.savefig(os.path.join(output, f'{folder_name}_grid.png'))


if __name__ == '__main__':
    parameters = ['tone_dist_loss', 'tone_dist_sigma', 'angles_weight']
    folder = './code/experiments/tone_dist_sigma'
    output = './code/experiments'
    make_grid(folder, output)

