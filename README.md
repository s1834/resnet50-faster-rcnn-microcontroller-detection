# resnet50-faster-rcnn-microcontroller-detection

Object detection project using ResNet-50 Faster R-CNN to identify microcontrollers in images.

---

## Setup

Install the required dependencies:

```bash
pip install torch torchvision albumentations opencv-python numpy matplotlib tqdm
```

---

## Dataset

1. Download the [Microcontroller Detection Dataset](https://www.kaggle.com/datasets/tannergi/microcontroller-detection).
2. Place the extracted folder in your project directory as:

```
Microcontroller Detection/
```

---

## Configuration

- Edit `config.py` to set your device.
  - Note: `cpu` is recommended for Apple Silicon (M1/M2) Macs.
- Model outputs will be saved in the `outputs/` folder.
- Predictions will be saved in the `test_predictions/` folder.

---

## Training

To train the model:

```bash
python engine.py
```

---

## Inference

To make predictions on test data:

```bash
python inference.py
```

---

## Notes

- Ensure the dataset is correctly placed as `Microcontroller Detection/`.
- All logs, checkpoints, and outputs will be organized automatically.

---
