import torch
import matplotlib.pyplot as plt
from config import *


def save_generated_images(num, images, generated):
    f_name = RESULTS_PATH + '{0:0=4d}.png'.format(num)
    fig = plt.figure(figsize=(18, 8))
    generated = torch.clamp(generated, min=0, max=1)

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.axis("off")
        plt.imshow(images[i].permute(1, 2, 0))

        plt.subplot(2, 5, i + 6)
        plt.axis("off")
        plt.imshow(generated[i].permute(1, 2, 0))

    plt.subplots_adjust(left=0.05,
                        bottom=0.05,
                        wspace=0.05,
                        hspace=0.05)
    plt.savefig(f_name)
    plt.close(fig)
