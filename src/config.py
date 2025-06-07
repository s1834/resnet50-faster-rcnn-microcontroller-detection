import torch

BATCH_SIZE = 4
RESIZE_TO = 512
NUM_EPOCHS = 100

DEVICE = torch.device("mps") if torch.mps.isavaailable() else torch.device("cpu")

TRAIN_DIR = '../Microcontroller Detection/train'

VALID_DIR = '../Microcontroller Detection/valid'

classes