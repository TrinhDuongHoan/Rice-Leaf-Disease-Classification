# Rice-Leaf-Disease-Classification

A deep learning project for rice leaf disease classification using MobileNetV3 with attention mechanisms (ECA, MHSA). Includes preprocessing, augmentation, training, and evaluation for multi-class image classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Web Application](#web-application)
- [Results](#results)

## 🎯 Overview

This project implements a deep learning solution for classifying rice leaf diseases into 10 categories. The system uses MobileNetV3 as the backbone with attention mechanisms (ECA, MHSA) to improve classification accuracy.

**Supported Disease Classes:**
1. Bacterial Leaf Blight
2. Bacterial Leaf Streak
3. Bacterial Panicle Blight
4. Blast
5. Brown Spot
6. Dead Heart
7. Downy Mildew
8. Hispa
9. Normal
10. Tungro

## ✨ Features

- 🚀 MobileNetV3 backbone with attention mechanisms (ECA/MHSA)
- 📊 Comprehensive data augmentation pipeline
- 📈 Training with early stopping and model checkpointing
- 🎨 Interactive web interface for real-time prediction
- 📉 Visualization tools for metrics and confusion matrix
- 💾 Best model saving

## 📊 Dataset

The Paddy Disease Classification dataset contains images of rice leaves with various diseases.

**Dataset Statistics:**
- Total images: 10,407
- Training set: ~80%
- Validation set: ~20%
- Image resolution: 224x224 (configurable)
- Number of classes: 10

**Dataset Structure:**
```
src/utils/data/Paddy_Dataset/
├── train_images/
│   ├── bacterial_leaf_blight/
│   ├── bacterial_leaf_streak/
│   ├── bacterial_panicle_blight/
│   ├── blast/
│   ├── brown_spot/
│   ├── dead_heart/
│   ├── downy_mildew/
│   ├── hispa/
│   ├── normal/
│   └── tungro/
├── test_images/
├── splits/
│   ├── train.csv
│   └── val.csv
├── train.csv
└── sample_submission.csv
```

## 🏗️ Model Architecture

### MobileNetV3 + ECA (Efficient Channel Attention)
- Lightweight backbone optimized for mobile devices
- ECA module for efficient channel-wise attention
- ~5M parameters

### MobileNetV3 + MHSA (Multi-Head Self-Attention)
- MobileNetV3 backbone
- MHSA mechanism for capturing long-range dependencies
- ~6M parameters

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Rice-Leaf-Disease-Classification.git
cd Rice-Leaf-Disease-Classification
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
torch
torchvision
timm
numpy
pandas
pyyaml
scikit-learn
tqdm
matplotlib
fastapi 
uvicorn 
pillow
python-multipart
```

### Step 4: Prepare Dataset
Make sure your dataset is placed in `src/utils/data/Paddy_Dataset/`

The dataset should have:
- `train_images/` folder with disease class subfolders
- `test_images/` folder
- `splits/train.csv` and `splits/val.csv` files

## 📁 Project Structure

```
Rice-Leaf-Disease-Classification/
├── src/
│   ├── main.py                  # Main training script
│   ├── app.py                   # Flask web application
│   │
│   ├── models/
│   │   ├── backbones/
│   │   │   └── mobilenet.py     # MobileNetV3 implementation
│   │   └── attentions/
│   │       ├── eca.py           # ECA attention module
│   │       └── mhsa.py          # MHSA attention module
│   │
│   ├── training/
│   │   └── train_model.py       # Training logic
│   │
│   ├── utils/
│   │   ├── data/
│   │   │   ├── dataset.py       # PyTorch Dataset
│   │   │   ├── loader.py        # DataLoader builder
│   │   │   ├── build_df_data.py # Data preprocessing
│   │   │   └── Paddy_Dataset/   # Dataset directory
│   │   │
│   │   └── metrics/
│   │       ├── benchmark.py     # Evaluation metrics
│   │       └── plots.py         # Visualization
│   │
│   ├── trained_model/           # Saved model weights
│   │   └── mobilenetv3_eca.pth
│   │
│   └── web/
│       ├── static/              # CSS, JS, uploaded images
│       └── templates/
│           └── index.html       # Web UI
│
└── README.md                    # This file
```

## 🚀 Usage

### Training

#### 1. Basic Training (Default Settings)
```bash
cd src
python main.py
```

This will train the model with default settings:
- Model: MobileNetV3 + ECA
- Batch size: 4
- Image size: 224x224
- Learning rate: 0.001
- Number of epochs: 50


#### 6. Available Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_csv` | str | `utils/data/Paddy_Dataset/splits/train.csv` | Path to training CSV |
| `--val_csv` | str | `utils/data/Paddy_Dataset/splits/val.csv` | Path to validation CSV |
| `--batch_size` | int | 4 | Batch size for training |
| `--num_workers` | int | 2 | Number of data loading workers |
| `--learning_rate` | float | 0.001 | Learning rate |
| `--num_epochs` | int | 50 | Number of training epochs |
| `--image_size` | int | 224 | Input image size |
| `--device` | str | cuda | Device (cuda/cpu) |


### Web Application

#### 1. Start the FastAPI Server
```bash
cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 2. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:8000
```
or
```
http://127.0.0.1:8000
```

**To access from other devices on the same network:**
```
http://YOUR_IP_ADDRESS:8000
```

#### 3. API Documentation (Swagger UI)
FastAPI automatically generates interactive API documentation:
```
http://localhost:8000/docs
```

**Alternative API documentation (ReDoc):**
```
http://localhost:8000/redoc
```

#### 4. Using the Web App

**Step-by-step:**
1. Navigate to `http://localhost:8000`
2. Click "**Choose File**" or "**Browse**" button
3. Select a rice leaf image from your computer (JPG, PNG, JPEG)
4. Click "**Upload**" or "**Predict**" button
5. View the prediction results:
   - Disease class name
   - Confidence score
   - Prediction probabilities for all classes
6. Upload another image to test more


## 📊 Results

### Model Performance

| Model | Accuracy | Parameters | Inference Time (GPU) |
|-------|----------|------------|---------------------|
| MobileNetV3 + ECA | ~92% | 5.4M | ~15ms |
| MobileNetV3 + MHSA | ~91% | 6.1M | ~18ms |

### Sample Predictions

The model can accurately classify various rice leaf diseases:
- ✅ Bacterial diseases (blight, streak, panicle blight)
- ✅ Fungal diseases (blast, brown spot, downy mildew)
- ✅ Pest damage (hispa, dead heart, tungro)
- ✅ Healthy leaves (normal)


### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Paddy Disease Classification Dataset](https://www.kaggle.com/competitions/paddy-disease-classification)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [ECA-Net Paper](https://arxiv.org/abs/1910.03151)
- [PyTorch](https://pytorch.org/)

---

⭐️ If you find this project helpful, please give it a star!