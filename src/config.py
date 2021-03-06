import torch


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# TRAIN_IMAGES_PATH = '..\\data\\edges2shoes\\edges2shoes\\train\\'
# VALID_IMAGES_PATH = '..\\data\\edges2shoes\\edges2shoes\\val\\'
TRAIN_IMAGES_PATH = '..\\data\\cars\\train\\'  # training dataset
VALID_IMAGES_PATH = '..\\data\\cars\\val\\'  # validation dataset
RESULTS_PATH = '..\\results\\'  # folder for results
DOODLES_PATH = '..\\data\\doodles\\'  # doodles for validation

IMAGE_SIZE = 256
BATCH_SIZE = 4
# LEARNING_RATE_DISCRIMINATOR = 5e-5
# LEARNING_RATE_GENERATOR = 1e-4
# NUM_EPOCHS = 15
LEARNING_RATE_DISCRIMINATOR = 5e-5
LEARNING_RATE_GENERATOR = 1e-4
NUM_EPOCHS = 40
