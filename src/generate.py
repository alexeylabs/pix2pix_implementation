import torch
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from config import *


def generate_images_from_doodles():
    generator = torch.load(RESULTS_PATH+'best_generator.pt')
    generator.eval()

    doodles_names = glob.glob(DOODLES_PATH+'*.png')
    with torch.no_grad():
        for doodle_name in doodles_names:
            doodle = np.array(Image.open(doodle_name))
            doodle = transforms.ToTensor()(doodle)
            doodle = TF.resize(doodle, (IMAGE_SIZE, IMAGE_SIZE))

            image = generator(doodle[None, :3, :, :].to(DEVICE))[0]
            image = transforms.ToPILImage()(image.cpu()).convert("RGB")

            f_name = 'image_from_' + doodle_name.split('\\')[-1]
            image.save(RESULTS_PATH+'doodles\\'+f_name, "JPEG")

            print(f_name, 'has saved.')


generate_images_from_doodles()

