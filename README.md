# Real-Time Age & Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository implements an advanced, dual-task computer vision pipeline designed to simultaneously predict a subject's **Age** and **Facial Emotion** in real-time. Built entirely on the PyTorch framework, the project leverages modern lightweight vision architectures and GPU-accelerated parallel inference to deliver high-accuracy continuous predictions at high frame rates.

---

## Architecture & Key Features

* **Dual-Task Inference**: Predicts 7 basic emotions (FER2013 standard) via classification and estimates exact age via linear regression.
* **Real-Time Pipeline**: Features an optimized webcam inference script utilizing **CUDA Streams** for asynchronous model execution, **FP16 (Half Precision)** for reduced VRAM footprint, and adaptive frame-skipping.
* **Temporal Stabilization**: Implements a custom Exponential Moving Average (EMA) algorithm to eliminate "age jittering" across consecutive frames, ensuring a temporally coherent UI overlay.
* **Robust Face Detection**: Integrates PyTorch-native **MTCNN** (`facenet-pytorch`) for precise facial bounding box extraction and alignment.
* **Modular Codebase**: Adheres to Object-Oriented principles, strictly separating configuration, dataset instantiation, training loops, and inference logic.

---

## Datasets & Preprocessing

The models were trained using distinct preprocessing pipelines tailored to their respective architectures. Dataset mapping and splits are managed via the CSV files located in the `metadata_splits/` directory, ensuring reproducible training and validation environments. 

* **Emotion Recognition**: Images are resized to 236x236 and center-cropped to 224x224.
* **Age Estimation**: Images undergo a high-resolution pipeline, resized and cropped to 384x384 to preserve fine morphological features (e.g., wrinkles, skin texture).

---

## Models & Performance

The system utilizes two state-of-the-art lightweight architectures. The following table summarizes the validation metrics achieved during the evaluation phase:

| Task | Architecture | Input Resolution | Primary Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Emotion** | `ConvNeXt-Tiny` | 224x224 | Accuracy / F1 (Happy) | ~73.0% / 0.88 |
| **Age** | `EfficientNetV2-S` | 384x384 | MAE / R² Score | 4.97 years / 0.86 |

*Note: The age regression model demonstrates high accuracy for standard demographics. A custom clipping mechanism and EMA stabilization are applied during live inference to handle boundary conditions and extreme edge cases.*

---

## Repository Structure

```text
.
├── checkpoints/                # Directory for saved model weights (.pth) - *Ignored in Git*
├── metadata_splits/            # CSV mapping files for Train/Validation datasets
├── notebooks/                  
│   ├── data_exploration.ipynb  # Exploratory Data Analysis (EDA)
│   └── model_eval.ipynb        # Confusion matrices, classification reports, and scatter plots
├── src/                        # Core Source Code
│   ├── config.py               # Global hyperparameters and path definitions
│   ├── dataset.py              # PyTorch Dataset classes and transformations
│   ├── face_stabilizer.py      # Temporal smoothing logic (EMA)
│   ├── model.py                # Architecture instantiations
│   └── trainer.py              # Training loops and evaluation metrics
├── demo.py                     # MAIN SCRIPT: Real-time webcam inference application
├── train.py                    # CLI script for model training
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/apiantonio/Age-Emotion_Recognition
cd age-emotion_recognition

```

### 2. Environment Configuration

It is highly recommended to use an isolated environment (e.g., Conda):

```bash
conda create -n facesight python=3.10
conda activate facesight

```

### 3. Install Dependencies

Ensure PyTorch is installed with the appropriate CUDA tooling for your hardware. Proceed with the remaining requirements:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python facenet-pytorch pandas numpy matplotlib seaborn scikit-learn tqdm

```

### 4. Download Pre-Trained Weights

Model checkpoints (`.pth` files) are hosted in the **GitHub Releases** section to comply with Git file size limits.

1. Navigate to the [Releases page](https://www.google.com/search?q=../../releases/latest) of this repository.
2. Download the pre-trained model weights located under the "Assets" section.
3. Place the downloaded files in the appropriate directories within `checkpoints/`. The structure must reflect the paths defined in the `MODELS_CONFIG` dictionary inside `demo.py`:
```text
checkpoints/
├── emotion_best/
│   └── best_model.pth
└── age_best/
    └── best_model.pth

```



---

## Usage Guide

### Live Inference (Webcam Demo)

To execute the real-time application with bounding box overlay and simultaneous predictions:

```bash
python demo.py

```

*Note: Press the `Q` key while the video window is in focus to terminate the process safely.*

### Model Training

The `train.py` CLI script facilitates training from scratch.

**To train the Emotion Classification model:**

```bash
python train.py --task emotion --model convnext_tiny --batch_size 64 --epochs 50

```

**To train the Age Regression model:**

```bash
python train.py --task age --model efficientnet_v2_s --batch_size 32 --epochs 50

```

*Best performing models and periodic checkpoints are automatically serialized into timestamped directories within `checkpoints/`.*

### Performance Evaluation

To generate classification reports, confusion matrices, and regression scatter plots, run the provided evaluation notebook:

```bash
jupyter notebook notebooks/model_eval.ipynb

```

---

## Technical Implementations

* **CUDA Streams**: Within `demo.py`, `torch.cuda.Stream()` enables concurrent execution of both the Emotion and Age models on the GPU, effectively mitigating inference bottlenecks.
* **Mixed Precision (FP16)**: Model parameters and input tensors are cast to half-precision (`.half()`), maximizing Tensor Core utilization on modern NVIDIA GPUs and significantly increasing framerates without accuracy degradation.
* **Stateful Face Tracking**: To prevent the regression model from yielding volatile frame-to-frame estimates, the `FaceStabilizer` class maps facial coordinates continuously using Euclidean distance thresholds, applying an Exponential Moving Average (EMA) to the raw regression outputs.

---

## License

This project is licensed under the MIT License.
