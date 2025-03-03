# 🚀 eventCamera_remoteTrain

## 📌 项目介绍
该项目提供了一套完整的CIFAR10-DVS的时间堆叠表示方法在mobileNetV2中的wandB中sweep搜索的远程训练流程

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
├── notebooks/
│   └── train.ipynb       (Jupyter 训练代码)
├── download_data.py      (自动下载数据集)
├── requirements.txt      (环境依赖)
├── train.py              (训练脚本)
├── .gitignore            (忽略大文件)
├── README.md             (项目说明)

