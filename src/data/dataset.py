import random
import numpy as np
from PIL import Image
import glob
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from config import *


class ImageDataset(Dataset):
    def __init__(self, images_path, use_augmentation=True):
        self.filenames = glob.glob(images_path + '*.jpg')
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        raw_image = np.array(Image.open(self.filenames[idx]))

        image = raw_image[:, :256, :]
        target = raw_image[:, 256:, :]

        image = transforms.ToTensor()(image)
        target = transforms.ToTensor()(target)

        image = TF.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        target = TF.resize(target, (IMAGE_SIZE, IMAGE_SIZE))

        if self.use_augmentation:
            # random mirroring and jitter
            if random.random() > 0.5:
                image = TF.vflip(image)
                target = TF.vflip(target)

            if random.random() > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)

            image = TF.adjust_brightness(image, 0.5 + random.random())
            image = TF.adjust_contrast(image, 0.5 + random.random())
            image = TF.adjust_saturation(image, 0.5 + random.random())
            image = TF.adjust_hue(image, random.random() - 0.5)

        return image, target
