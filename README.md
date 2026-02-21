# Real-Time Age & Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CDB.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E.svg)
![PWA](https://img.shields.io/badge/PWA-Ready-5A0FC8.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository implements an advanced, dual-task computer vision pipeline designed to simultaneously predict a subject's **Age** and **Facial Emotion** in real-time. Built entirely on the PyTorch framework, the project leverages modern lightweight vision architectures and GPU-accelerated parallel inference to deliver high-accuracy continuous predictions at high frame rates.

---

## Architecture & Key Features

* **Dual-Task Inference**: Predicts 7 basic emotions (FER2013 standard) via classification and estimates exact age via linear regression.
* **Real-Time Pipeline**: Features an optimized webcam inference script utilizing **CUDA Streams** for asynchronous model execution, **FP16 (Half Precision)** for reduced VRAM footprint, and adaptive frame-skipping.
* **Temporal Stabilization**: Implements a custom Exponential Moving Average (EMA) algorithm to eliminate "age jittering" across consecutive frames, ensuring a temporally coherent UI overlay.
* **Robust Face Detection**: Integrates PyTorch-native **MTCNN** (`facenet-pytorch`) for precise facial bounding box extraction and alignment.
* **Modular Codebase**: Adheres to Object-Oriented principles, strictly separating configuration, dataset instantiation, training loops, and inference logic.
* **Lightweight Face Detection**: The computationally expensive MTCNN algorithm was replaced with **`face-api.js`**, specifically utilizing its **SSD (Single Shot Multibox Detector) MobileNet V1** architecture. This transition ensures robust, scale-invariant multi-face tracking with a minimal memory footprint, preventing thermal throttling and event-loop blocking on resource-constrained mobile browsers.
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

## Web Application (PWA) Demo

To facilitate immediate testing without requiring a local PyTorch environment or GPU hardware, a Progressive Web App (PWA) has been implemented. The application is hosted via GitHub Pages and executes the entire computer vision pipeline directly within the browser, optimized for mobile performance.

* **Live Demo**: [FaceSight Web App](https://apiantonio.github.io/Age-Emotion_Recognition/)
### Web Models Pipeline

| Task | Library / Framework | Architecture | Execution Provider | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Face Detection** | `face-api.js` | [Tiny Face Detector](https://justadudewhohacks.github.io/face-api.js/docs/index.html#models-face-detection) | WASM (CPU) | Float32 |
| **Emotion** | `onnxruntime-web` | ConvNeXt-Tiny | WASM (XNNPACK) | INT8 (Quantized) |
| **Age** | `onnxruntime-web` | EfficientNetV2-S | WebGPU / WASM | Float32 |

### Web Architecture & Limitations

The web application addresses the hardware constraints of mobile devices through a high-performance, multi-threaded architecture:

* **Multi-Threaded Web Worker**: To ensure a fluid User Interface at 60 FPS, the entire AI pipeline (detection, preprocessing, and inference) has been moved to a dedicated Web Worker (ai-worker.js). This prevents the main thread from freezing during complex calculations, allowing the video stream and UI animations to remain perfectly smooth.
* **Static INT8 Quantization (Emotion)**: The Emotion Recognition model has been optimized via Static Quantization. By converting weights from Float32 to INT8, the model size was reduced by ~75% with negligible accuracy loss, significantly speeding up inference on mobile CPUs via the WASM XNNPACK backend.
* **Hybrid Execution Strategy**: The application employs a selective execution strategy to balance speed and precision:
    * **Emotion (INT8)**: Runs on the CPU via WASM, as integer math is highly optimized for mobile processors.
    * **Age (Float32)**: Leverages WebGPU for massive parallelization on the device's graphics card, maintaining maximum regression accuracy without taxing the CPU.
* **High-Resolution Age Pipeline**: Unlike the emotion task (224x224), the age estimation utilizes a 384x384 resolution to preserve fine morphological features (wrinkles, skin texture) necessary for precise estimation.
* **Optimized Memory Management**: The pipeline utilizes Tensors directly for face detection to avoid DOM-related bottlenecks within the Worker environment and implements immediate memory disposal (.dispose()) to prevent RAM saturation during continuous real-time usage.
* **Fully Offline Capable**: A dedicated Service Worker caches the UI assets and the .onnx model weights locally. Once the initial 100MB+ load is complete, the app functions entirely offline, behaving like a native mobile application.
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
├── webapp/                     # Progressive Web App (PWA) Source Code
│   ├── index.html              # UI and layout
│   ├── script.js               # Face-API and ONNX Runtime inference logic
│   ├── sw.js                   # Service Worker for offline caching and network routing
│   └── manifest.json           # PWA installation manifest
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

Due to Git file size limits, model weights are hosted externally. The required files depend on the environment you intend to run.

#### A. Desktop / Local PyTorch Inference (`.pth` weights)
The standard PyTorch checkpoints are hosted in the **GitHub Releases** section.

1. Navigate to the [Releases page](../../releases/latest) of this repository.
2. Download the pre-trained `.pth` model weights located under the "Assets" section.
3. Place the downloaded files in the appropriate directories within `checkpoints/`. The structure must reflect the paths defined in the `MODELS_CONFIG` dictionary inside `demo.py`:
```text
checkpoints/
├── emotion_best/
│   └── best_model.pth
└── age_best/
    └── best_model.pth
```
#### B. Web Application Inference (.onnx weights)
For the client-side WebAssembly execution, the models have been exported to the ONNX format.

GitHub Releases: The .onnx files are archived alongside the .pth files in the GitHub Releases for reference.

Hugging Face (Active CDN): To bypass browser CORS policies and ensure reliable global delivery, the web application actively fetches the ONNX weights hosted on a dedicated Hugging Face dataset. No manual download is required to run the live PWA, as the JavaScript pipeline handles the remote fetching automatically.

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
