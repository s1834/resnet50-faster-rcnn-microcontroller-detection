import torch

BATCH_SIZE = 4
RESIZE_TO = 512
NUM_EPOCHS = 100

# DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
DEVICE = torch.device("cpu")

TRAIN_DIR = '../Microcontroller Detection/train'

VALID_DIR = '../Microcontroller Detection/valid'

CLASSES = ['background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora']

NUM_CLASSES = 5

VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2

