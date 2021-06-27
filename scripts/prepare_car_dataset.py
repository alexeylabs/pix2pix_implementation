# конвертация датасета
# https://www.kaggle.com/c/carvana-image-masking-challenge/data
# для задачи pix2pix

from PIL import Image
import numpy as np
import cv2 as cv
from skimage import color
import glob
from tqdm import tqdm


PHOTOS_PATH = 'cars\\train\\'
MASK_PATH = 'cars\\train_masks\\'
SAVE_PATH = 'cars\\results\\'

file_names = glob.glob(PHOTOS_PATH+'*.jpg')
file_names = [fname.split('.')[0].split('\\')[-1] for fname in file_names]

for fname in tqdm(file_names):
    mask = Image.open(MASK_PATH+fname+'_mask.gif')
    mask = np.array(mask)

    img = Image.open(PHOTOS_PATH+fname+'.jpg')
    img = np.array(img)
    img = np.where(mask[:, :, None] == 1, img, 255)

    # img = img[:, 319:319+1280, :]
    img = np.pad(img, ((319, 319), (0, 0), (0, 0)), mode='constant', constant_values=255)

    img = cv.resize(img, (256, 256))

    edges = cv.Canny(img, 100, 200)
    edges = np.where(edges > 0, 0, 255)
    edges = color.gray2rgb(edges)

    res = np.hstack([edges, img])

    Image.fromarray(res.astype(np.uint8)).save(SAVE_PATH+fname+'.jpg')