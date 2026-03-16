# DeepLens PINN: Gravitational Lensing Dark Matter Classification

## Project Overview

This project implements machine learning models for classifying gravitational lensing images to identify dark matter substructure types. The work is part of the **ML4Sci Google Summer of Code** initiative and focuses on applying deep learning and physics-informed neural networks (PINNs) to astronomical image analysis under the project **DeepLense**.

The dataset contains 150×150 pixel grayscale images of gravitational lenses, classified into three categories:
- **Class 0**: No substructure
- **Class 1**: Spherical/subhalo substructure
- **Class 2**: Vortex substructure

## Dataset
In the notebooks, we use X_train.npy, X_test.npy, y_train.npy and y_test.npy
Those files are created by running the `build_complete_dataset.py` script on the dataset given by the evaluators,
which can be found at [dataset.zip - Google Drive](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view).

To create those files, extract dataset.zip and modify the `TRAINING_DATA_PATH` and `TEST_DATA_PATH` in `build_complete_dataset.py` and run the script values accordingly to obtain the files.
```bash
    python3 build_complete_dataset.py
```

Alternatively, I have created a public Kaggle dataset and Google Drive links to access those files:
- [Kaggle Dataset](https://www.kaggle.com/datasets/abhirajraje/lensing-data)
- [Google Drive Link](https://drive.google.com/drive/folders/13R0pBp1ChEvVH049ODjI6T09zwCcNRKz?usp=sharing)

## Requirements

- Python 3.14+
- PyTorch >= 1.9.0
- NumPy
- Matplotlib
- Scikit-learn
- tqdm
- Kaggle(optional, for direct dataset download)

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Deeplens_PINN
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn tqdm
   ```

## Outputs

- **Plots**: 
  - `sample_images.png` - Sample images from each class
  - ROC-AUC curves (displayed in notebook)
  - Training progress plots

- **Models**:
  - `best-model-specific-task.pt` - Trained model weights of specific task PINN model
  - `best-model-common-task.pt` - Trained model weights of common task ResNet

