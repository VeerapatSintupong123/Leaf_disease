# Leaf Disease Classification

This repository contains code for training and evaluating a deep learning model to classify leaf diseases using PyTorch and torchvision. The project uses a ResNet18 backbone for image classification and provides visualization with Grad-CAM.

## Features

- **Data Preparation**: Loads and processes image data and labels from CSV files.
- **Model**: Uses a modified ResNet18 architecture for 4-class classification.
- **Training**: Includes training and validation loops with learning rate finder and cosine annealing scheduler.
- **Evaluation**: Computes accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization**: Uses Grad-CAM for model interpretability and displays predictions with probabilities.

## Folder Structure

```
Leaf_disease/
├── data/
│   ├── images/
│   └── train.csv
├── checkpoints/
│   └── tf_resnet18_15_1.pth
├── train_2.ipynb
├── rust.jpg
├── scab.jpg
├── lotus.jpg
├── cabbage.jpg
├── rust2.jpg
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- tqdm
- Pillow
- pytorch-grad-cam

Install dependencies with:
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn seaborn tqdm pillow pytorch-grad-cam
```

## Usage

1. **Prepare Data**: Place your images in `data/images/` and ensure `data/train.csv` is formatted as expected.
2. **Train Model**: Run the notebook `train_2.ipynb` to train and evaluate the model.
3. **Checkpoints**: Model weights are saved in the `checkpoints/` directory.
4. **Visualization**: Grad-CAM visualizations and predictions are shown for sample images at the end of the notebook.

## Example

After training, you can visualize predictions and Grad-CAM heatmaps for new images:

```python
img_path = "rust.jpg"
predicted, probs = get_leaf_disease(
    img_path=img_path,
    model=model,
    transforms=transforms,
    show=True
)
for label, prob in zip(classes, probs):
    print(f"{label}: {prob:.3f}")
```

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---