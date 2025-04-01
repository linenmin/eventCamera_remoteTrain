
# 🚀 eventCamera_remoteTrain

## 📌 Project Overview

This project is part of my master's thesis, which focuses on deploying neural networks based on event cameras to the embedded platform **KV260**. The code in this repository implements a remote training pipeline using **TensorFlow 2** and **MobileNetV2**, with integrated hyperparameter tuning via [Weights & Biases (wandb)](https://wandb.ai/).

The dataset is stored on Google Drive, and the code includes scripts for downloading, training, and hyperparameter sweeps. The training data, `timeStack_data_1281281`, is derived from **CIFAR10-DVS** using a time-stacking approach, with the input dimensions of **128×128×3**.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/linenmin/eventCamera_remoteTrain.git
cd eventCamera_remoteTrain
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

```bash
python download_data.py
```

### 4. Unzip the Dataset

```bash
unzip data/timeStack_data_1281281.zip -d data/
```

### 5. Configure Wandb API Key

**Linux/macOS:**

```bash
export WANDB_API_KEY=your_wandb_api_key
```

**Windows:**

```bash
set WANDB_API_KEY=your_wandb_api_key
```

### 6. Run the Sweep Script

```bash
python run_sweep.py
```

The best-performing model will be saved under the `models/` directory, with filenames like:

```
models/besttimeStack1281281_<learning_rate>_<batch_size>_<dropout_rate>_<l2_reg>_mbNetV2.h5
```

---

## 🧩 Dataset Conversion Tool

The script `timeStack_data_transform.py` is provided as a standalone tool to convert the **CIFAR10-DVS** dataset from its original `.aedat` format into **TFRecord** format, making it compatible with TensorFlow pipelines.

### Features:
- Parses `.aedat` files from CIFAR10-DVS
- Applies **time stacking** to create 128×128×3 event image representations
- Outputs `.tfrecord` files for training

> **Note:** This preprocessing step is essential to prepare event-based data for training neural networks and supports the embedded deployment workflow.

---

## 📁 Project Structure

```
eventCamera_remoteTrain/
├── data/                      # Unzipped dataset (not included in repo)
├── models/                    # Trained models will be saved here
├── download_data.py           # Script to download the dataset
├── requirements.txt           # Python dependencies
├── tf2_mbNetV2_train.py       # Training logic using MobileNetV2
├── run_sweep.py               # Wandb hyperparameter sweep script
├── timeStack_data_transform.py# CIFAR10-DVS AEDAT to TFRecord converter
├── .gitignore                 # Excludes large files and datasets
├── README.md                  # Project documentation
```

---

Feel free to cite or fork this project if you're working on event-based vision systems or embedded AI!
