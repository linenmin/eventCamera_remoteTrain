# 🚀 eventCamera_remoteTrain

## 📌 项目介绍
本项目提供了一套完整的远程训练流程，代码托管在 GitHub，数据集存储在 Google Drive。

用户可以在本地或 Google Colab 上运行本项目，并自动下载数据集进行训练。

---

## 🛠️ 环境安装
### 1️⃣ 克隆仓库
```bash
git clone https://github.com/linenmin/eventCamera_remoteTrain.git
cd eventCamera_remoteTrain
```

### 2️⃣ 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 3️⃣ 下载数据集
```bash
python download_data.py
```

### 4️⃣ 解压数据
```bash
unzip data/timeStack_data_1281281.zip -d data/
```

### 5️⃣ 运行训练脚本
```bash
python train.py
```

---

## 🚀 Google Colab 快速开始
点击下方链接，在 Google Colab 运行：

[![在 Colab 中运行](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linenmin/eventCamera_remoteTrain/blob/main/notebooks/train.ipynb)

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
```

---

## 📜 许可证
本项目遵循 MIT 许可证，详情请查看 [LICENSE](LICENSE) 文件。

---

## ❓ 常见问题
### 1️⃣ **数据下载失败怎么办？**
如果 `python download_data.py` 失败，可能是 Google Drive 链接变更或网络问题。
请手动下载 [数据集](https://drive.google.com/file/d/1BI1idNJeow8zTftjzP7Dud-ixXXRqjJQ/view?usp=sharing)，然后放入 `data/` 目录。

### 2️⃣ **Colab 运行时报错怎么办？**
请确保 **Colab 运行时启用了 GPU**（`Runtime -> Change runtime type -> GPU`）。

### 3️⃣ **如何贡献代码？**
1. Fork 本仓库。
2. 创建新分支进行修改。
3. 提交 PR（Pull Request）。

---

如果有其他问题，请提交 [Issue](https://github.com/linenmin/eventCamera_remoteTrain/issues) 反馈！😊

