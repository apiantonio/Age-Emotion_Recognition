# Real-Time Age & Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repo is an advanced, dual-task computer vision pipeline designed to simultaneously predict a person's **Age** and **Facial Emotion** in real-time. Built entirely on PyTorch, this project leverages modern lightweight vision architectures and GPU-accelerated parallel inference to deliver high-accuracy results at high frame rates.

---

## Key Features

* **Dual-Task AI**: Predicts 7 basic emotions (FER2013 standard) and estimates exact age via regression.
* **Real-Time Performance**: Optimized webcam demo utilizing **CUDA Streams** for parallel model execution, **FP16 (Half Precision)**, and adaptive frame-skipping.
* **Temporal Stabilization**: Exponential Moving Average eliminates "age jittering" across consecutive frames, providing a smooth and professional UI overlay.
* **Robust Face Detection**: Uses PyTorch-native **MTCNN** (`facenet-pytorch`) for highly accurate face detection and cropping.
* **Modular Codebase**: Clean, object-oriented design separating configuration, data loading, training loops, and inference.

---

## Models & Performance

The system utilizes two distinct State-of-the-Art lightweight architectures, each fine-tuned for its specific task with dedicated preprocessing pipelines (distinct resize/crop dimensions).

### 1. Emotion Recognition
* **Architecture**: `ConvNeXt-Tiny`
* **Input Size**: 224x224 (Resized from 236)
* **Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
* **Results**: Achieved **~73.0% Accuracy** on the validation set, with F1-Scores peaking at **0.88 for Happy** and **0.83 for Surprise**, competing with SOTA benchmarks for lightweight models on standard "in-the-wild" datasets.

### 2. Age Estimation
* **Architecture**: `EfficientNetV2-S`
* **Input Size**: 384x384
* **Task**: Linear Regression (1 Output Node)
* **Results**: 
  * **MAE (Mean Absolute Error)**: `4.97 years`
  * **R² Score**: `0.86`
  * *Note: The model is highly accurate for standard demographics, with a custom clipping and EMA stabilization applied during live inference to handle extreme edge cases.*

---

## 📂 Repository Structure

```text
.
├── checkpoints/                # Directory for saved model weights (.pth) - *Ignored in Git*
├── metadata_splits/            # CSV mapping files for Train/Validation splits
├── notebooks/                
│   ├── data_exploration.ipynb  # EDA on datasets
│   └── model_eval.ipynb        # Confusion matrices, classification reports, and scatter plots
├── src/                        # Core Source Code
│   ├── config.py               # Global hyperparameters and path definitions
│   ├── dataset.py              # PyTorch Datasets and transformations
│   ├── face_stabilizer.py      # Temporal smoothing logic (EMA) for age predictions
│   ├── model.py                # Model instantiation (ConvNeXt, EfficientNet)
│   └── trainer.py              # Training loops, loss functions, and metrics tracking
├── demo.py                     # MAIN SCRIPT: Real-time webcam inference
├── train.py                    # CLI script to train models from scratch
├── .gitignore                  # Git ignore rules for large files
└── README.md                   # Project documentation

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/apiantonio/Age-Emotion_Recognition
cd age-emotion_recognition

```

### 2. Create a Virtual Environment (Recommended)

Using Conda:

```bash
conda create -n facesight python=3.10
conda activate facesight

```

### 3. Install Dependencies

Make sure you have PyTorch installed with CUDA support (if you have an NVIDIA GPU). Then install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python facenet-pytorch pandas numpy matplotlib seaborn scikit-learn tqdm

```

### 4. Download Pre-Trained Weights
Model checkpoints (`.pth` files) are hosted in the **GitHub Releases** section.

1. Go to the [Releases page](../../releases/latest) of this repository.
2. Download the pre-trained model weights (e.g., the `.pth` files under "Assets").
3. Place them in the appropriate folders inside the `checkpoints/` directory. Based on the default configuration, the structure should look like this:
   ```text
   checkpoints/
   ├── emotion_best/
   │   └── best_model.pth
   └── age_best/
       └── best_model.pth
Ensure the paths defined in the MODELS_CONFIG dictionary inside demo.py match exactly where you placed the downloaded files.

---

## Usage

### Running the Real-Time Demo

To launch the webcam application with real-time bounding boxes and predictions:

```bash
python demo.py

```

*Press `Q` while the video window is focused to safely exit the application.*

### Training the Models

You can train the models from scratch using the `train.py` CLI script.

**Train the Emotion Model:**

```bash
python train.py --task emotion --model convnext_tiny --batch_size 64 --epochs 50

```

**Train the Age Regression Model:**

```bash
python train.py --task age --model efficientnet_v2_s --batch_size 32 --epochs 50

```

*Checkpoints and best models will be automatically saved in timestamped directories inside `checkpoints/`.*

### Evaluating Performance

To visualize confusion matrices, F1-scores, and Age Regression scatter plots, launch Jupyter Notebook and open:

```bash
jupyter notebook notebooks/model_eval.ipynb

```

---

## Technical Highlights

* **CUDA Streams**: In `demo.py`, PyTorch `torch.cuda.Stream()` is used to execute the Emotion and Age models simultaneously on the GPU, halving the inference bottleneck.
* **Mixed Precision (FP16)**: Model weights and input tensors are cast to half-precision (`.half()`), maximizing tensor core utilization on modern RTX GPUs and boosting FPS.
* **Face Stabilizer**: To prevent the regression model from outputting violently flickering age estimates, a custom class tracks faces across frames via Euclidean distance and applies an Exponential Moving Average (EMA).

---
