# 🚀 eventCamera_remoteTrain

## 📌 项目介绍
本项目基于 TensorFlow 2 和 MobileNetV2 构建了一个用于远程训练的模型，并集成了 wandb 超参搜索功能。数据集存储在 Google Drive，代码中提供了数据下载、训练和超参搜索（sweep）的完整流程。
数据timeStack_data_1281281通过CIFAR10-DVS的时间堆叠数据表示转化而来,维度是128*128*3

---

## 🛠️ 环境安装
### 1. 克隆仓库
```bash
git clone https://github.com/linenmin/eventCamera_remoteTrain.git
cd eventCamera_remoteTrain
```

### 2。 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 3. 下载数据集
```bash
python download_data.py
```

### 4. 解压数据
```bash
unzip data/timeStack_data_1281281.zip -d data/
```

### 5. 配置 Wandb API 密钥
```bash
export WANDB_API_KEY=你的_wandb_api_key
```
或者在 Windows 下：
```bash
set WANDB_API_KEY=你的_wandb_api_key
```

### 6. 运行sweep脚本
```bash
python run_sweep.py
```

---
---

## 📦 项目结构
```
eventCamera_remoteTrain/
├── data/                 (不会上传到 GitHub)
│   └── (存放解压后的数据)
├── models/                 
│   └── (存放最佳模型)
├── download_data.py      (自动下载数据集)
├── requirements.txt      (环境依赖)
├── tf2_mbNetV2_train.py  (训练定义)
├── run_sweep.py          (搜索训练脚本)
├── .gitignore            (忽略数据集)
├── README.md             (项目说明)

